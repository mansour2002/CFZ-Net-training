

import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import albumentations as A

import config
import utils
import model
import dataset # For get_valid_transforms

def preprocess_image(image_path, img_size, transforms):
    """Loads, resizes, and preprocesses a single image for inference."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
    image = image / 255.0

    transformed = transforms(image=image)
    image = transformed['image']
    
    image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float().unsqueeze(0) # Add batch dimension
    return image

def postprocess_mask(prediction_tensor, class_colors):
    """Converts a predicted mask tensor to a colored numpy image."""
    prediction = torch.argmax(prediction_tensor.squeeze(0), dim=0).cpu().numpy() # Remove batch dim and get class index
    
    colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(class_colors):
        colored_mask[prediction == class_idx] = color
    return colored_mask

def predict_single_image(model, image_path, img_size, class_colors, model_load_path=None):
    """Performs inference on a single image and returns the colored segmentation mask."""
    device = utils.get_device()
    model.to(device)

    if model_load_path:
        model.load_state_dict(torch.load(model_load_path, map_location=device))
    
    model.eval()

    # Use the same transforms as validation, but without mask
    inference_transforms = dataset.get_valid_transforms(img_size)

    image_tensor = preprocess_image(image_path, img_size, inference_transforms)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
    
    colored_mask = postprocess_mask(output, class_colors)
    return colored_mask

if __name__ == '__main__':
    # Example usage:
    # Ensure you have a trained model saved as "best_model.pth" in config.SAVE_PATH
    # And a test image path
    
    # Path to your test image (replace with an actual image path)
    test_image_path = os.path.join(config.DATASET_PATH, 'train/Input', 'test_image.png') # Update this!

    # Initialize model
    cfz_model = model.CFZNet(classes=config.NUM_OF_CLASSES)
    
    # Path to the saved model weights
    model_weights_path = os.path.join(config.SAVE_PATH, "best_model.pth")

    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights not found at {model_weights_path}. Please train the model first.")
    elif not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}. Please provide a valid path.")
    else:
        print(f"Performing prediction on: {test_image_path}")
        predicted_mask = predict_single_image(
            cfz_model,
            test_image_path,
            config.IMAGE_SIZE,
            config.CLASS_COLORS,
            model_weights_path
        )
        
        # Save or display the predicted mask
        output_filename = "predicted_mask.png"
        output_path = os.path.join(config.SAVE_PATH, output_filename)
        cv2.imwrite(output_path, predicted_mask)
        print(f"Predicted mask saved to: {output_path}")

        # You can also display it
        # plt.imshow(predicted_mask)
        # plt.title("Predicted Segmentation Mask")
        # plt.axis('off')
        # plt.show()