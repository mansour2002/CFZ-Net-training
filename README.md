# CFZ-Net-training

Deep Learning for Capillary Free Zones (CFZ) Segmentation using OCTA Images. The implementation uses Python and PyTorch.

## Overview
This GitHub repository hosts the implementation of CFZ-Net, a deep learning model developed in PyTorch for segmenting capillary free zones (CFZ), arteries, and veins in optical coherence tomography angiography (OCTA) images. The model is trained on high-resolution 6x6mm OCTA images, aiming to accurately delineate vascular structures for medical research and diagnosis.

## Key Features:
- Utilizes PyTorch framework for efficient training and inference
- Designed specifically for segmenting CFZ, arteries, and veins in OCTA images
- Preprocessing scripts for data augmentation and preparation
- Evaluation metrics for assessing segmentation performance


Images were acquired using the AngioVue SD-OCT device (Optovue, Fremont, CA, USA). The OCT system had a 70,000 Hz A-scan rate with ~5 μm axial and ~15 μm lateral resolutions. All OCTA images used for this study were 6 mm × 6 mm scans; only superficial OCTA images were used.

## Figures
Here are two representative OCTA images and corresponding manually generated ground truths and predicted images:
![The CFZ-Net](https://github.com/mansour2002/CFZ-Net-training/blob/main/figures/CFZ%20Segmentation%201.png?raw=true)


![The CFZ-Net](https://github.com/mansour2002/CFZ-Net-training/blob/main/figures/CFZ%20Segmentation%202.png?raw=true)


## Repository Structure

```
CFZ-Net-training/
├── src/
│   ├── config.py       # Configuration settings (paths, hyperparameters, classes)
│   ├── utils.py        # Utility functions (device handling, data splitting, mask encoding)
│   ├── dataset.py      # PyTorch Dataset class and data augmentation transforms
│   ├── model.py        # CFZ-Net model definition
│   ├── train.py        # Script for training the model
│   └── predict.py      # Script for performing inference on new images
├── data/               # Directory for raw and processed datasets
│   └── (e.g., Dataset/train/Input, Dataset/train/CFZ_map)
├── models/             # Directory to save trained model weights
│   └── (e.g., best_model.pth)
├── figures/            # Directory for project figures and visualizations
│   └── (e.g., CFZ Segmentation 1.png)
├── notebooks/          # Original Jupyter notebooks
│   └── CFZ-Net.ipynb
├── requirements.txt    # Python dependencies
└── README.md           # Project README file
```


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mansour2002/CFZ-Net-training.git
    cd CFZ-Net-training
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation

* **Update `config.py`** (if needed):
    * `PARENT_DIR` is automatically set to the repository root directory.
    * Set `SOURCE_DATA_DIR` to the path where your raw 6x6mm OCTA images are located (if you need to process raw data).

* **Run data transfer (if necessary)**:
    The `utils.transfer_data` function in `utils.py` will read your raw data, resize it, normalize OCTA images, create a 3-channel OCTA image, and save them to the `data/train/Input` and `data/train/CFZ_map` directories. It will also create a CSV file (`train_data_tmp_Idea61CFZ.csv`) mapping input to output images. This function is commented out by default in `train.py` because it only needs to be run once.

    To process raw data:
    1. Update `SOURCE_DATA_DIR` in `config.py` to point to your raw data location
    2. Uncomment the data transfer section in `src/train.py` (lines 124-131)
    3. Run `python src/train.py` once

### 2. Training the Model

To train the CFZ-Net model, run:
```bash
python src/train.py
```
* You can adjust training parameters (e.g., `BATCH_SIZE`, `CURRENT_FOLD`, `NUM_EPOCHS` in the train.py file) in [src/config.py](src/config.py).
* The best model weights (based on validation loss) will be saved in the `models/` directory as `best_model.pth`.

### 3. Making Predictions

To perform inference on a single image, run:
```bash
python src/predict.py
```
* **Before running**:
    * Update `test_image_path` in [src/predict.py](src/predict.py) (line 68) to the actual path of the image you want to segment.
    * Ensure you have a trained model at `models/best_model.pth`.
* The script will load the best-trained model and save the predicted segmentation mask to the `models/` directory as `predicted_mask.png`.

## Dependencies

* Python >= 3.9
* PyTorch >= 2.2.1
* CUDA >= 11.8 (for GPU acceleration, optional)
* torchvision >= 0.17.0
* torchmetrics >= 1.0.0
* segmentation-models-pytorch >= 0.3.0
* albumentations >= 1.3.0
* opencv-python >= 4.8.0
* numpy >= 1.24.0
* pandas >= 2.0.0
* Pillow >= 10.0.0
* matplotlib >= 3.7.0

Please refer to [requirements.txt](requirements.txt) for the complete list of dependencies.

