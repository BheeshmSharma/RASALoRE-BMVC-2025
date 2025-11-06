## 🧠 RASALoRE: Region Aware Spatial Attention with Location-based Random Embeddings
<div align="center">

**Weakly Supervised Anomaly Detection in Brain MRI Scans**

- [![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/pdf/2510.08052)
<!-- 
- [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
- [![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
- [![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)
-->

*A novel two-stage weakly supervised framework for accurate brain anomaly detection with minimal computational overhead*

</div>

## 📋 Table of Contents
- [🎯 Abstract](#-abstract)
- [🔄 Updates](#-updates)
- [🔧 Environment Setup](#-environment-setup)
- [📊 Datasets](#-datasets)
- [⚡ Quick Start](#-quick-start)
- [🚀 Running RASALoRE](#-running-rasalore)
- [📝 Citation](#-citation)
- [📄 License](#-license)

## 🎯 Abstract

Weakly Supervised Anomaly detection (WSAD) in brain MRI scans is an important challenge useful to obtain quick and accurate detection of brain anomalies, when precise pixel-level anomaly annotations are unavailable and only weak labels (e.g., slice-level) are available. In this work, we propose RASALoRE: Region Aware Spatial Attention with Location-based Random Embeddings, a novel two-stage WSAD framework. In the first stage, we introduce a Discriminative Dual Prompt Tuning (DDPT) mechanism that generates high-quality pseudo weak masks based on slice-level labels, serving as coarse localization cues. In the second stage, we propose a segmentation network with a region-aware spatial attention mechanism that relies on fixed location-based random embeddings. This design enables the model to effectively focus on anomalous regions. Our approach achieves state-of-the-art anomaly detection performance, significantly outperforming existing WSAD methods while utilizing less than 8 million parameters. Extensive evaluations on the BraTS20, BraTS21, BraTS23, and MSD datasets demonstrate a substantial performance improvement coupled with a significant reduction in computational complexity.

<div align="center">
  <img src="/Figures/RASALoRE_Arch.png" alt="RASALoRE Architecture" width="800"/>
  <p><em>RASALoRE Framework Architecture</em></p>
</div>


## 🔄 Updates

> **📢 Latest announcements and project milestones**

- **🆕 [2025-11-20]** - RASALoRE code release
- **🆕 [2025-11-20]** - DDPT code release
- **🆕 [2025-11-20]** - Initial code release and repository setup
- **📝 [2025-07-25]** - Paper accepted at BMVC 2025

<!-- 
- **🚀 [2025-11-17]** - Repository created and initial commit
- **🎯 [2024-12-01]** - Pre-trained weights for all datasets available
- **⚡ [2024-11-25]** - Performance benchmarks published
- **🔧 [2024-11-20]** - Environment setup and documentation completed


---

### 📅 Upcoming Updates
- [ ] Pre-trained weights
- [ ] Multi-Modality Extension  
-->


## 🔧 Environment Setup

### Prerequisites

* **Python** 3.8 or higher
* **CUDA-compatible GPU**
* **Conda** package manager
* **Minimum 12 GB GPU memory** required for training with default settings

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/BheeshmSharma/RASALoRE-BMVC-2025.git
cd RASALoRE

# 2. Create and activate conda environment
conda env create -f environment.yml
conda activate rasa

# 3. Install PyTorch via conda
conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. Verify installation
python -c "import torch; print('✅ PyTorch installed successfully')"
```

## 📊 Datasets

Our framework supports multiple medical imaging datasets:

| Dataset | Description | Download Link |
|---------|-------------|---------------|
| **BraTS20** | Brain Tumor Segmentation Challenge 2020 | [🔗 Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) |
| **BraTS21** | Brain Tumor Segmentation Challenge 2021 | [🔗 Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data) |
| **BraTS23** | Brain Tumor Segmentation Challenge 2023 | [🔗 Synapse](https://www.synapse.org/Synapse:syn51156910/wiki/621282) |
| **MSD** | Medical Segmentation Decathlon | [🔗 Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) |

<!-- 
| **MSLUB** | Multiple Sclerosis Dataset (Ljubljana) | [🔗 Official Site](https://lit.fe.uni-lj.si/en/research/resources/3D-MR-MS/) |
-->

### 📁 Data Preprocessing

```bash
# 1. Download Dataset
Download your chosen dataset from the links provided above

# 2. Configure Dataset Path
nano ./DATA/Data_Extraction.py
# Update the dataset path in `dataset_root` variable to point to your downloaded data

# 3. Extract and Process Data
cd DATA
python Data_Extraction.py

# 4. Verify Data Structure
# Check that the processed data matches the expected format
# Refer to DDPT/readme.md for detailed structure requirements
```

## ⚡ Quick Start

```bash
# 1. Generate DDPT Masks
Navigate to `DDPT/` and follow the [detailed instructions](./DDPT/readme.md)

# 2. Verify DDPT masks are properly generated and saved
ls ./DATA/{Dataset Name} 

# 3. Run MedSAM inference with DDPT guidance
cd MedSAM
python MedSAM_Inference_DDPT-Guided.py --Dataset {Dataset Name}

```

## 🚀 Running RASALoRE

### 🔧 Pre-training Steps

#### Fixed Candidate Location Embeddings

```bash
cd Fixed_Candidate_Embeddings
python Fixed_Candidate_Location_Embedding.py
```

> This script generates and saves the fixed candidate location prompt embeddings.

##### ✅ Notes:

* After running the script, verify that the embedding file `Candidate_Prompt_Embedding.pt` has been generated in the `Fixed_Candidate_Embeddings/` directory.
* You can generate multiple random embeddings by re-running the script.
* ⚠️ **Important:** Ensure that you use the **same embedding file** for both training and testing to maintain consistency and prevent evaluation mismatches.

### 🏃‍♂️ Training

Configure training parameters in `train_run.py`:
- Dataset selection
- Number of epochs
- Learning rate
- Early stopping criteria
- Other hyperparameters

```bash
python train_run.py
```

### 🧪 Testing

Configure testing parameters in `test_run.py`:
- Dataset
- Model checkpoint path
- Inference settings
- ⚠️ **Important:** Ensure that you use the **same random embedding** for testing that was used for training.
  
```bash
python test_run.py
```

<!-- 
### 💾 Pre-trained Model Weights

| Dataset | Model Weights |
|---------|---------------|
| BraTS20 | [📥 To Be Released] |
| BraTS21 | [📥 To Be Released] |
| BraTS23 | [📥 To Be Released] |
| MSD | [📥 To Be Released] |
-->

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{
anonymous2025rasalore,
title={{RASAL}o{RE}: Region Aware Spatial Attention with Location-based Random Embeddings for Weakly Supervised Anomaly Detection in~Brain~{MRI}~Scans},
author={Anonymous},
booktitle={The Thirty Sixth British Machine Vision Conference},
year={2025},
url={https://openreview.net/forum?id=A2Kylh60ai}
}
```

## 🙏 Acknowledgments
- We gratefully acknowledge Technocraft Centre of Applied Artificial Intelligence (TCAAI), IIT Bombay, for their support through generous funding.

<!-- 
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
-->

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/BheeshmSharma/RASALoRE-BMVC-2025.svg?style=social&label=Star)](https://github.com/BheeshmSharma/RASALoRE-BMVC-2025)
[![GitHub forks](https://img.shields.io/github/forks/BheeshmSharma/RASALoRE-BMVC-2025.svg?style=social&label=Fork)](https://github.com/BheeshmSharma/RASALoRE-BMVC-2025/fork)

</div>
