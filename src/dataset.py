
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

import utils
import config

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, tfms, label_colors_list, classes_to_train, all_classes):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = utils.set_class_values(self.all_classes, self.classes_to_train)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        image = image / 255.0
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        mask = utils.get_label_mask(mask, self.class_values, self.label_colors_list)

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).long()

        return image, mask

def get_train_transforms(img_size):
    """Transforms/augmentations for training images and masks."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(shear=0.4, mode=4, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1.0),
        A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=50, shift_limit=0.3, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True),
        A.Blur(blur_limit=3, p=0.2),
    ])

def get_valid_transforms(img_size):
    """Transforms/augmentations for validation images and masks."""
    return A.Compose([A.Resize(img_size, img_size, always_apply=True)])