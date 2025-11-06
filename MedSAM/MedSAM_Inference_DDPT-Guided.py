import os
import cv2 
import torch
import random
import argparse
import requests
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import SamModel, SamProcessor

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='MedSAM Inference with DDPT Prompt')
    parser.add_argument('--Dataset', type=str, required=True, 
                       help='Dataset name (e.g., BraTS20)')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load MedSAM model
print("Loading MedSAM model...")
MedSAM = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(device)
processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
MedSAM.eval()
print("Model loaded successfully!")

# Function to generate random points inside a bounding box and round them
def generate_random_points(bbox, num_points=10, decimal_places=0):
    x_min, y_min, x_max, y_max = bbox
    points = []
    labels = []  # Corresponding labels for the points
    for _ in range(num_points):
        # Generate random x and y coordinates inside the bounding box
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        
        # Round the coordinates to the specified number of decimal places
        points.append([round(x, decimal_places), round(y, decimal_places)])
        
        # Assign label '1' for all points (you can modify this if needed)
        labels.append(1)
    
    return points, labels

# Function to get bounding box
def get_bounding_box(ground_truth_map):
    ground_truth_map = np.array(ground_truth_map)
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:  # If there are no non-zero elements
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # Add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    
    return bbox

# Function to get bounding boxes for a batch
def get_bounding_boxes_for_batch(batch_masks):
    bounding_boxes = []
    for mask in batch_masks:
        bbox = get_bounding_box(mask)
        if bbox:
            bounding_boxes.append(bbox)
    return bounding_boxes

# Configuration
Dataset_name = args.Dataset
Prompt_type = "Box"  # Default prompt type (Box, PointsPrompt)

print(f"Processing dataset: {Dataset_name}")
print(f"Using prompt type: {Prompt_type}")

# Setup paths
image_folder_path = f'../DATA/{Dataset_name}/Training_images/Unhealthy/'
DDPT_mask_folder_path = f'../DATA/{Dataset_name}/Training_images/Maps/'
directory = f'../DATA/{Dataset_name}/Training_images/MedSAM_Mask_with_DDPT_Prompt_{Prompt_type}/'

# Check if input directories exist
if not os.path.exists(image_folder_path):
    raise FileNotFoundError(f"Image folder not found: {image_folder_path}")
if not os.path.exists(DDPT_mask_folder_path):
    raise FileNotFoundError(f"DDPT mask folder not found: {DDPT_mask_folder_path}")

# Create output directory
os.makedirs(directory, exist_ok=True)
print(f"Output directory: {directory}")

# Get list of files
Filenames = os.listdir(image_folder_path)
print(f"Found {len(Filenames)} files to process")

count = 0.0
for filename in tqdm(Filenames, desc="Processing images"):
    try:
        image_path = os.path.join(image_folder_path, filename)
        image = Image.open(image_path).convert("RGB")

        image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        corrected_array = np.power(image_array, 2.0)  # Apply gamma correction
        image = [Image.fromarray((corrected_array*255).astype(np.uint8))]  # Rescale to [0, 255]
        
        # Load and normalize the DPT mask image
        DDPT_mask_path = os.path.join(DDPT_mask_folder_path, filename)
        if not os.path.exists(DDPT_mask_path):
            print(f"Warning: DDPT mask not found for {filename}, skipping...")
            continue
            
        DDPT_mask_image = cv2.imread(DDPT_mask_path, cv2.IMREAD_GRAYSCALE)
        DDPT_mask_image = (DDPT_mask_image - DDPT_mask_image.min()) / (DDPT_mask_image.max() - DDPT_mask_image.min())
        DDPT_mask_image = np.where(DDPT_mask_image > 0.5, 1.0, 0.0)

        # Get bounding boxes for the batch
        input_boxes = get_bounding_box(DDPT_mask_image)
        if input_boxes is None:
            print(f"Warning: No valid bounding box found for {filename}, skipping...")
            continue
            
        # Transform the list of bounding boxes
        transformed_boxes = [[input_boxes]]
        
        if Prompt_type == "Box":
            # Prepare inputs for the model
            inputs = processor(image, input_boxes=[transformed_boxes], return_tensors="pt").to(device)
        elif Prompt_type == "PointsPrompt":
            points, labels = generate_random_points(input_boxes)  # Generate 10 points and labels for each bounding box
            # Prepare inputs for the model
            inputs = processor(image, input_points=[points], input_labels=[labels], return_tensors="pt").to(device)
        else:
            raise ValueError(f"Unknown prompt type: {Prompt_type}")

        # Predict
        with torch.no_grad():
            outputs = MedSAM(**inputs, multimask_output=False)

        preds = torch.sigmoid(outputs.pred_masks.squeeze(1)).squeeze(1)
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        preds = torch.where(preds > 0.5, 1.0, 0.0).squeeze(0)

        numpy_array = preds.cpu().numpy()
        result_image = Image.fromarray(numpy_array.astype(np.uint8)*255)
        result_image.save(os.path.join(directory, filename))
        
        count += 1
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

print(f"Successfully processed {count} images")
print(f"Results saved in: {directory}")