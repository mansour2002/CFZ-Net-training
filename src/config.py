

import os

# Base directory for the project
PARENT_DIR = 'D:\\AV Project\\CFZ-Net'  # IMPORTANT: Update this path to your project root

# Folder to save models, logs, and other outputs
SAVE_FOLDER_NAME = "CFZ_segmentation_V5T02"
SAVE_PATH = os.path.join(PARENT_DIR, SAVE_FOLDER_NAME)

# DataFrame CSV file name
DATAFRAME_NAME = 'train_data_tmp_Idea61CFZ.csv'

# Project-specific paths
PROJECT_PATH = os.path.join(PARENT_DIR, 'tmp_Idea61CFZ')
DATASET_PATH = os.path.join(PROJECT_PATH, 'Dataset')
TRAIN_INPUT_PATH = os.path.join(DATASET_PATH, 'train/Input')
TRAIN_OUTPUT_PATH = os.path.join(DATASET_PATH, 'train/CFZ_map')

# Image size for resizing
IMAGE_SIZE = 320

# Classes and their corresponding colors
CLASSES = ['background', 'vCFZ', 'vein', 'aCFZ', 'artery']
NUM_OF_CLASSES = len(CLASSES)
CLASS_COLORS = [(0, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0)]

# Training parameters
BATCH_SIZE = 15
FOLDS = 5 # Total number of folds for cross-validation (if used)
CURRENT_FOLD = 1 # Current fold number (if using a specific fold)

# Source directory for raw data (if data transfer is needed)
SOURCE_DATA_DIR = "D:\\AV Project\\Dataset6x6mm\\"

# Ensure directories exist
def create_directories():
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(TRAIN_INPUT_PATH, exist_ok=True)
    os.makedirs(TRAIN_OUTPUT_PATH, exist_ok=True)

create_directories()