

import os

# Base directory for the project
# IMPORTANT: Update this path to your project root
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder to save models, logs, and other outputs
SAVE_FOLDER_NAME = "models"
SAVE_PATH = os.path.join(PARENT_DIR, SAVE_FOLDER_NAME)

# DataFrame CSV file name
DATAFRAME_NAME = 'train_data_tmp_Idea61CFZ.csv'

# Project-specific paths
PROJECT_PATH = os.path.join(PARENT_DIR, 'data')
DATASET_PATH = PROJECT_PATH
TRAIN_INPUT_PATH = os.path.join(DATASET_PATH, 'train', 'Input')
TRAIN_OUTPUT_PATH = os.path.join(DATASET_PATH, 'train', 'CFZ_map')

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
# IMPORTANT: Update this path to your raw data location
SOURCE_DATA_DIR = os.path.join(PARENT_DIR, 'raw_data')

# Ensure directories exist
def create_directories():
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(TRAIN_INPUT_PATH, exist_ok=True)
    os.makedirs(TRAIN_OUTPUT_PATH, exist_ok=True)

create_directories()