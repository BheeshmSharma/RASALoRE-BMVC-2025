import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import random

# Configuration constants
STABILITY_MARGIN = 15  # Frames to skip at volume edges
TRAIN_SPLIT = 0.80   # Training data percentage
IMAGE_DIMS = (240, 240)  # Standard brain scan dimensions

# Modality mapping for different BraTS versions
MODALITY_MAPPING = {
    'BraTS20': {
        't1': 't1',
        't1ce': 't1ce', 
        't2': 't2',
        'flair': 'flair'
    },
    'BraTS21': {
        't1': 't1',
        't1ce': 't1ce',
        't2': 't2', 
        'flair': 'flair'
    },
    'BraTS23': {
        't1': 't1n',      # t1-native
        't1ce': 't1c',    # t1-contrast enhanced
        't2': 't2w',      # t2-weighted
        'flair': 't2f'    # t2-flair
    },
    'MSD': {
        't1': 1,          # T1w is at index 1
        't1ce': 2,        # t1gd (T1 with gadolinium) is at index 2
        't2': 3,          # T2w is at index 3
        'flair': 0        # FLAIR is at index 0
    }
}

def detect_dataset_version(dataset_folder):
    """Detect whether the dataset is BraTS20, BraTS21, BraTS23, or MSD based on folder structure"""
    
    # Check for MSD structure first
    if ('imagesTr' in os.listdir(dataset_folder) and 
        'labelsTr' in os.listdir(dataset_folder)):
        return 'MSD'
    
    # Check for BraTS structures
    patient_folders = [f for f in os.listdir(dataset_folder) 
                      if os.path.isdir(os.path.join(dataset_folder, f))]
    
    if not patient_folders:
        raise ValueError("No patient folders found in the dataset directory")
    
    # Check the first folder to determine dataset version
    first_folder = patient_folders[0]
    
    if first_folder.startswith('BraTS20_Training_'):
        return 'BraTS20'
    elif first_folder.startswith('BraTS2021_'):
        return 'BraTS21'
    elif first_folder.startswith('BraTS-GLI-'):
        return 'BraTS23'
    else:
        raise ValueError(f"Unknown dataset format. First folder: {first_folder}")

def get_dataset_modality(dataset_version, requested_modality):
    """Get the correct modality name/index for the specific dataset version"""
    if dataset_version not in MODALITY_MAPPING:
        raise ValueError(f"Unknown dataset version: {dataset_version}")
    
    if requested_modality not in MODALITY_MAPPING[dataset_version]:
        available_modalities = list(MODALITY_MAPPING[dataset_version].keys())
        raise ValueError(f"Modality '{requested_modality}' not available for {dataset_version}. "
                        f"Available modalities: {available_modalities}")
    
    return MODALITY_MAPPING[dataset_version][requested_modality]

def get_file_paths(patient_path, patient_folder, modality, dataset_version):
    """Get the correct file paths based on dataset version"""
    if dataset_version == 'MSD':
        # For MSD, we don't need modality-specific paths since all modalities are in one file
        # patient_folder here is just the patient ID like "BRATS_001"
        volume_path = os.path.join(patient_path, f"{patient_folder}.nii.gz")
        seg_path = volume_path.replace('imagesTr', 'labelsTr')  # Labels are in parallel directory
        return volume_path, seg_path
    else:
        # Get the correct modality name for this dataset version
        actual_modality = get_dataset_modality(dataset_version, modality)
        
        if dataset_version == 'BraTS20':
            volume_file = f"{patient_folder}_{actual_modality}.nii"
            seg_file = f"{patient_folder}_seg.nii"
        elif dataset_version == 'BraTS21':
            volume_file = f"{patient_folder}_{actual_modality}.nii.gz"
            seg_file = f"{patient_folder}_seg.nii.gz"
        elif dataset_version == 'BraTS23':
            volume_file = f"{patient_folder}-{actual_modality}.nii.gz"
            seg_file = f"{patient_folder}-seg.nii.gz"
        else:
            raise ValueError(f"Unknown dataset version: {dataset_version}")
        
        volume_path = os.path.join(patient_path, volume_file)
        seg_path = os.path.join(patient_path, seg_file)
        
        return volume_path, seg_path

def save_frame_direct(volume_frame, mask_frame, frame_index, volume_name, 
                     healthy_folder, unhealthy_folder, mask_folder):
    """Save individual frame directly to appropriate folder based on mask values"""
    # Convert the volume and mask frames to images
    volume_img = Image.fromarray(volume_frame.astype(np.uint8))
    mask_img = Image.fromarray(mask_frame.astype(np.uint8))

    # Construct filenames
    base_filename = f"{volume_name}_{frame_index}.png"
    mask_filename = os.path.join(mask_folder, base_filename)

    # Determine if frame is healthy or unhealthy and save accordingly
    non_zero_pixels = np.count_nonzero(mask_frame)
    
    if non_zero_pixels > 200:
        volume_filename = os.path.join(unhealthy_folder, base_filename)
        volume_img.save(volume_filename)
        mask_img.save(mask_filename)  # Save mask only for unhealthy frames
        return "unhealthy"
    elif non_zero_pixels == 0:
        # Healthy frame - save both image and mask
        volume_filename = os.path.join(healthy_folder, base_filename)
        volume_img.save(volume_filename)
        mask_img.save(mask_filename)  # Save mask only for healthy frames
        return "healthy"

def get_all_patients(dataset_folder, dataset_version):
    """Get all patient IDs from the dataset folder, works for all supported versions"""
    if dataset_version == 'MSD':
        # For MSD, get patient IDs from imagesTr folder
        images_folder = os.path.join(dataset_folder, 'imagesTr')
        if not os.path.exists(images_folder):
            raise ValueError(f"imagesTr folder not found in {dataset_folder}")
        
        patient_files = [f for f in os.listdir(images_folder) if f.endswith('.nii.gz')]
        # Remove the .nii.gz extension to get patient IDs
        patient_ids = [f.replace('.nii.gz', '') for f in patient_files]
        return patient_ids
    else:
        # For BraTS datasets, get patient folders
        patient_folders = []
        for f in os.listdir(dataset_folder):
            if os.path.isdir(os.path.join(dataset_folder, f)):
                if (f.startswith('BraTS20_Training_') or 
                    f.startswith('BraTS2021_') or 
                    f.startswith('BraTS-GLI-')):
                    patient_folders.append(f)
        return patient_folders

def process_brats_dataset_direct(dataset_folder, train_healthy_folder, train_unhealthy_folder,
                               test_healthy_folder, test_unhealthy_folder, mask_folder, 
                               train_patients, test_patients, modality='t1ce'):
    """Process BraTS dataset (20, 21, 23, or MSD) and save directly to final folders"""
    
    # Detect dataset version
    dataset_version = detect_dataset_version(dataset_folder)
    print(f"Detected dataset version: {dataset_version}")
    
    # Validate modality for this dataset version
    try:
        modality_info = get_dataset_modality(dataset_version, modality)
        if dataset_version == 'MSD':
            print(f"Using modality: {modality} -> index {modality_info}")
        else:
            print(f"Using modality: {modality} -> {modality_info}")
    except ValueError as e:
        print(f"Error: {e}")
        return {}
    
    # Ensure all output directories exist
    os.makedirs(train_healthy_folder, exist_ok=True)
    os.makedirs(train_unhealthy_folder, exist_ok=True)
    os.makedirs(test_healthy_folder, exist_ok=True)
    os.makedirs(test_unhealthy_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # Counters for summary
    counters = {
        'train_healthy': 0,
        'train_unhealthy': 0,
        'test_healthy': 0,
        'test_unhealthy': 0,
        'ambiguous': 0
    }

    # Get list of all patients
    all_patients = get_all_patients(dataset_folder, dataset_version)
    
    for patient_id in tqdm(all_patients, desc="Processing patients"):
        
        # Determine if this patient is in train or test set
        if patient_id in train_patients:
            current_healthy_folder = train_healthy_folder
            current_unhealthy_folder = train_unhealthy_folder
            split_type = "train"
        elif patient_id in test_patients:
            current_healthy_folder = test_healthy_folder
            current_unhealthy_folder = test_unhealthy_folder
            split_type = "test"
        else:
            print(f"Warning: Patient {patient_id} not found in train or test sets")
            continue
        
        # Get file paths based on dataset version
        try:
            if dataset_version == 'MSD':
                images_folder = os.path.join(dataset_folder, 'imagesTr')
                labels_folder = os.path.join(dataset_folder, 'labelsTr')
                volume_path = os.path.join(images_folder, f"{patient_id}.nii.gz")
                seg_path = os.path.join(labels_folder, f"{patient_id}.nii.gz")
            else:
                patient_folder_path = os.path.join(dataset_folder, patient_id)
                volume_path, seg_path = get_file_paths(patient_folder_path, patient_id, modality, dataset_version)
        except ValueError as e:
            print(f"Error getting file paths for {patient_id}: {str(e)}")
            continue
        
        # Check if both files exist
        if not os.path.exists(volume_path):
            print(f"Warning: Volume file not found: {volume_path}")
            continue
        if not os.path.exists(seg_path):
            print(f"Warning: Segmentation file not found: {seg_path}")
            continue
        
        # Load the volume and segmentation
        try:
            volume_nii = nib.load(volume_path)
            volume_data = volume_nii.get_fdata()
            
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata()
            
            # Handle MSD's 4D structure (extract specific modality)
            if dataset_version == 'MSD':
                if len(volume_data.shape) == 4:
                    # Extract the specific modality from 4D volume
                    modality_index = get_dataset_modality(dataset_version, modality)
                    volume_data = volume_data[:, :, :, modality_index]
                else:
                    print(f"Warning: Expected 4D volume for MSD dataset, got {volume_data.shape}")
                    continue
            
            # Extract patient name
            patient_name = patient_id
            
            # Iterate through the frames (3rd dimension of the volume)
            # Skip edge frames for stability
            for frame_index in range(STABILITY_MARGIN, volume_data.shape[2] - STABILITY_MARGIN):
                volume_frame = volume_data[:, :, frame_index]
                seg_frame = seg_data[:, :, frame_index]
                
                # Normalize volume frame to 0-255 range
                volume_frame = ((volume_frame - volume_frame.min()) / 
                              (volume_frame.max() - volume_frame.min() + 1e-8) * 255)
                
                # Convert segmentation to binary (0 or 255)
                seg_frame_binary = (seg_frame > 0).astype(np.uint8) * 255
                
                # Save frame directly to appropriate folder
                result = save_frame_direct(volume_frame, seg_frame_binary, frame_index, patient_name,
                                         current_healthy_folder, current_unhealthy_folder, mask_folder)
                
                # Update counters
                if result == "healthy":
                    counters[f'{split_type}_healthy'] += 1
                elif result == "unhealthy":
                    counters[f'{split_type}_unhealthy'] += 1
                else:
                    counters['ambiguous'] += 1
                         
        except Exception as e:
            print(f"Error processing {patient_id}: {str(e)}")
            continue
    
    return counters

def split_patients(patient_ids, train_ratio=TRAIN_SPLIT):
    """Splits patient IDs into train and test sets"""
    random.seed(42)  # For reproducibility
    patient_ids_copy = patient_ids.copy()  # Create a copy to avoid modifying original
    random.shuffle(patient_ids_copy)  # Shuffle for randomness

    total = len(patient_ids_copy)
    train_idx = int(train_ratio * total)

    train_patients = set(patient_ids_copy[:train_idx])
    test_patients = set(patient_ids_copy[train_idx:])

    return train_patients, test_patients

def print_available_modalities(dataset_version):
    """Print available modalities for the detected dataset version"""
    if dataset_version in MODALITY_MAPPING:
        modalities = list(MODALITY_MAPPING[dataset_version].keys())
        if dataset_version == 'MSD':
            print(f"\nAvailable modalities for {dataset_version}:")
            modality_names = {0: 'FLAIR', 1: 'T1w', 2: 't1gd', 3: 'T2w'}
            for std_name in modalities:
                index = MODALITY_MAPPING[dataset_version][std_name]
                actual_name = modality_names[index]
                print(f"  - {std_name} (maps to index {index}: {actual_name})")
        else:
            actual_names = list(MODALITY_MAPPING[dataset_version].values())
            print(f"\nAvailable modalities for {dataset_version}:")
            for std_name, actual_name in zip(modalities, actual_names):
                print(f"  - {std_name} (maps to {actual_name})")
    else:
        print(f"Unknown dataset version: {dataset_version}")

def process_brats_pipeline_direct(dataset_root, output_root, modality='t1ce'):
    """Complete pipeline for BraTS dataset processing with direct saving (works for BraTS20, BraTS21, BraTS23, and MSD)"""
    
    print("Step 1: Detecting dataset version and getting all patient IDs...")
    # Detect dataset version
    try:
        dataset_version = detect_dataset_version(dataset_root)
        print(f"Dataset version detected: {dataset_version}")
        print_available_modalities(dataset_version)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Validate modality
    try:
        modality_info = get_dataset_modality(dataset_version, modality)
        if dataset_version == 'MSD':
            print(f"Requested modality '{modality}' maps to index {modality_info} in {dataset_version}")
        else:
            print(f"Requested modality '{modality}' maps to '{modality_info}' in {dataset_version}")
    except ValueError as e:
        print(f"Error: {e}")
        print_available_modalities(dataset_version)
        return
    
    # Get all patient IDs
    all_patients = get_all_patients(dataset_root, dataset_version)
    print(f"Total patients found: {len(all_patients)}")
    
    if len(all_patients) == 0:
        print("No patients found. Please check the dataset path.")
        return
    
    print("Step 2: Splitting patients into train and test sets...")
    # Split patients into train and test
    train_patients, test_patients = split_patients(all_patients)
    
    print(f"Training patients: {len(train_patients)}")
    print(f"Testing patients: {len(test_patients)}")
    
    print("Step 3: Processing dataset and saving directly to final folders...")
    # Define paths for final folders
    train_healthy_folder = os.path.join(output_root, "Training_images", "Healthy")
    train_unhealthy_folder = os.path.join(output_root, "Training_images", "Unhealthy")
    test_healthy_folder = os.path.join(output_root, "Testing_images", "Healthy")
    test_unhealthy_folder = os.path.join(output_root, "Testing_images", "Unhealthy")
    mask_folder = os.path.join(output_root, "Masks")
    
    # Process dataset and save directly
    counters = process_brats_dataset_direct(
        dataset_root, train_healthy_folder, train_unhealthy_folder,
        test_healthy_folder, test_unhealthy_folder, mask_folder,
        train_patients, test_patients, modality
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Dataset Version: {dataset_version}")
    if dataset_version == 'MSD':
        print(f"Modality: {modality} -> index {modality_info}")
    else:
        print(f"Modality: {modality} -> {modality_info}")
    print(f"Training Data:")
    print(f"  - Healthy frames: {counters['train_healthy']}")
    print(f"  - Unhealthy frames: {counters['train_unhealthy']}")
    print(f"Testing Data:")
    print(f"  - Healthy frames: {counters['test_healthy']}")
    print(f"  - Unhealthy frames: {counters['test_unhealthy']}")
    print(f"Ambiguous frames (masks saved only): {counters['ambiguous']}")
    print(f"\nTotal frames processed: {sum(counters.values())}")
    print(f"All corresponding masks saved in: {mask_folder}")
    print("Applied standard image preprocessing and artifact filtering")

if __name__ == "__main__":
    # Configuration - You can easily switch between BraTS20, BraTS21, BraTS23, and MSD
    dataset_root = "Path for BraTS20 Dataset"
    output_root = "./BraTS20"  # "./BraTS20", "./BraTS21", "./BraTS23", "./MSD"
    
    
    # Available modalities (use standard names, they will be automatically mapped):
    # 't1', 't1ce', 't2', 'flair'
    modality = "t2"  
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Run the complete pipeline
    process_brats_pipeline_direct(dataset_root, output_root, modality)