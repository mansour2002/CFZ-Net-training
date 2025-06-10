

import os
import torch
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile

import config

def get_device():
    """Return the appropriate device (CUDA or CPU) based on availability."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(x):
    """Send the tensor or model to the appropriate device."""
    device = get_device()
    return x.to(device)

def get_model_parameters(model):
    """Returns the total number of trainable parameters in a model."""
    return sum(param.numel() for param in model.parameters())

def set_class_values(all_classes, classes_to_train):
    """Assigns a specific class label to each of the classes."""
    class_indices = {cls.lower(): idx for idx, cls in enumerate(all_classes)}
    class_values = [class_indices[cls.lower()] for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """Encodes image pixels into class-specific labels based on colors."""
    mask = np.array(mask)
    label_colors = np.array(label_colors_list)
    label_mask = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=int)

    for value, color in enumerate(label_colors):
        if value in class_values:
            matches = np.all(mask == color, axis=-1)
            label_mask[matches] = value
    return label_mask

def load_image_paths(df, path, column_index):
    """Loads image paths from a DataFrame based on a column index."""
    images = []
    for _, item in df.iterrows():
        temp_item = os.path.join(path, item.iloc[column_index])
        images.append(temp_item)
    return images

def transfer_data(source_dir, destination_dir, image_size, dataframe_name, parent_dir):
    """Transfers and processes image files from a source directory to a destination.
    Creates a CSV file with input/output paths."""
    files = []
    # Create necessary subdirectories if they don't exist
    os.makedirs(os.path.join(destination_dir, 'train/Input'), exist_ok=True)
    os.makedirs(os.path.join(destination_dir, 'train/CFZ_map'), exist_ok=True)

    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder)
        ava_map_path = os.path.join(subfolder_path, 'AVA_map')
        octa_path = os.path.join(subfolder_path, 'OCTA')

        if os.path.isdir(ava_map_path):
            for filename in os.listdir(ava_map_path):
                if filename.endswith('_CFZ_MM.png'):
                    base_filename = filename[:-11]
                    file_octa_tiff = os.path.join(octa_path, base_filename + '.tiff')
                    file_octa_png = os.path.join(octa_path, base_filename + '.png')
                    file_cfz_map = os.path.join(ava_map_path, filename)

                    # Check for .tiff first, then .png
                    if os.path.isfile(file_octa_tiff):
                        file_octa = file_octa_tiff
                    elif os.path.isfile(file_octa_png):
                        file_octa = file_octa_png
                    else:
                        print(f"OCTA image not found for {base_filename}")
                        continue

                    # Define the destination paths
                    output_png_filename = base_filename + '.png'
                    destination_input = os.path.join(destination_dir, 'train/Input', output_png_filename)
                    destination_output = os.path.join(destination_dir, 'train/CFZ_map', output_png_filename)

                    # Read and resize the OCTA and CFZ map images
                    im_octa = cv2.imread(file_octa, cv2.IMREAD_GRAYSCALE)
                    im_cfz_map = cv2.imread(file_cfz_map)

                    if im_octa is None or im_cfz_map is None:
                        print(f"Could not read images for {base_filename}")
                        continue

                    im_octa = cv2.resize(im_octa, (image_size, image_size))
                    im_cfz_map = cv2.resize(im_cfz_map, (image_size, image_size))

                    # Normalize OCTA images to 0-255 scale
                    im_octa = np.array(im_octa, dtype=np.float32)
                    im_cfz_map = np.array(im_cfz_map, dtype=np.float32)
                    
                    if np.max(im_octa) - np.min(im_octa) > 0:
                        im_octa = 255.0 * (im_octa - np.min(im_octa)) / (np.max(im_octa) - np.min(im_octa))
                    else:
                        im_octa = np.zeros_like(im_octa) # Handle cases where min == max

                    # Convert grayscale image to 3-channel
                    im_octa_3chs = cv2.merge([im_octa, im_octa, im_octa])

                    # Save the processed images
                    cv2.imwrite(destination_input, im_octa_3chs.astype(np.uint8))
                    cv2.imwrite(destination_output, im_cfz_map.astype(np.uint8))

                    files.append(output_png_filename)

    files = list(set(files)) # Ensure unique filenames
    dframe = pd.DataFrame({'Input': files, 'CFZ': files})
    dframe.to_csv(os.path.join(parent_dir, dataframe_name), index=False)
    print(f"Transferred {len(files)} image pairs and created {dataframe_name}")
    return files

def get_images(parent_dir, fold_num, max_fold, dataframe_name, input_path, output_path):
    """Loads paths for training and validation images and masks based on the fold."""
    train_csv_path = os.path.join(parent_dir, dataframe_name)
    df = pd.read_csv(train_csv_path)

    num_samples = len(df)
    start_val = int((fold_num - 1) / max_fold * num_samples)
    end_val = int(fold_num / max_fold * num_samples)

    df_train = pd.concat([df[:start_val], df[end_val:]], ignore_index=True)
    df_val = df[start_val:end_val]

    train_images = load_image_paths(df_train, input_path, 0)
    train_masks = load_image_paths(df_train, output_path, 1)
    valid_images = load_image_paths(df_val, input_path, 0)
    valid_masks = load_image_paths(df_val, output_path, 1)

    return train_images, train_masks, valid_images, valid_masks