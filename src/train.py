

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics as TM
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import config
import utils
import dataset
import model

# Initialize device
device = utils.get_device()
print(f"Using device: {device}")

def train_model(
    model,
    train_data_loader,
    valid_data_loader,
    optimizer,
    loss_fn,
    num_epochs,
    classes,
    save_path,
    model_name="best_model.pth"
):
    best_loss = float('inf')

    # Metrics
    metrics = TM.MetricCollection({
        'MulticlassAccuracy': TM.MulticlassAccuracy(num_classes=len(classes), average='macro', ignore_index=0),
        'MulticlassJaccardIndex': TM.MulticlassJaccardIndex(num_classes=len(classes), average='macro', ignore_index=0),
        'MulticlassPrecision': TM.MulticlassPrecision(num_classes=len(classes), average='macro', ignore_index=0),
        'MulticlassRecall': TM.MulticlassRecall(num_classes=len(classes), average='macro', ignore_index=0),
        'MulticlassF1Score': TM.MulticlassF1Score(num_classes=len(classes), average='macro', ignore_index=0)
    }).to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_data_loader:
            inputs, targets = utils.to_device(inputs), utils.to_device(targets)
            targets = targets.squeeze(1) # Remove channel dimension from mask

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_data_loader.dataset)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_data_loader:
                inputs, targets = utils.to_device(inputs), utils.to_device(targets)
                targets = targets.squeeze(1) # Remove channel dimension from mask

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs, dim=1)
                metrics.update(preds, targets)

        val_epoch_loss = val_running_loss / len(valid_data_loader.dataset)
        computed_metrics = metrics.compute()
        metrics.reset()

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
        for metric_name, metric_value in computed_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # Save the best model
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_path, model_name))
            print(f"Saved best model with Val Loss: {best_loss:.4f}")

    print("Training complete.")

def visualize_batch(inputs, targets, predictions=None, num_images=3):
    """Visualizes a batch of input images, ground truth masks, and optional predictions."""
    plt.figure(figsize=(15, 5 * num_images))
    for i in range(min(num_images, inputs.shape[0])):
        # Input Image
        plt.subplot(num_images, 3, i*3 + 1)
        plt.imshow(inputs[i].cpu().numpy().transpose(1, 2, 0))
        plt.title('Input Image')
        plt.axis('off')

        # Ground Truth Mask
        plt.subplot(num_images, 3, i*3 + 2)
        # Convert target mask to color for visualization
        colored_target_mask = np.zeros((*targets.shape[2:], 3), dtype=np.uint8)
        for class_idx, color in enumerate(config.CLASS_COLORS):
            colored_target_mask[targets[i].cpu().numpy().squeeze() == class_idx] = color
        plt.imshow(colored_target_mask)
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Predicted Mask
        if predictions is not None:
            plt.subplot(num_images, 3, i*3 + 3)
            # Convert prediction mask to color for visualization
            colored_pred_mask = np.zeros((*predictions.shape[2:], 3), dtype=np.uint8)
            for class_idx, color in enumerate(config.CLASS_COLORS):
                colored_pred_mask[predictions[i].cpu().numpy().squeeze() == class_idx] = color
            plt.imshow(colored_pred_mask)
            plt.title('Predicted Mask')
            plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Data preparation
    # Call data transfer function only if needed (e.g., first run)
    # utils.transfer_data(
    #     config.SOURCE_DATA_DIR,
    #     config.DATASET_PATH,
    #     config.IMAGE_SIZE,
    #     config.DATAFRAME_NAME,
    #     config.PARENT_DIR
    # )

    train_images, train_masks, valid_images, valid_masks = utils.get_images(
        config.PARENT_DIR,
        config.CURRENT_FOLD,
        config.FOLDS,
        config.DATAFRAME_NAME,
        config.TRAIN_INPUT_PATH,
        config.TRAIN_OUTPUT_PATH
    )

    train_tfms = dataset.get_train_transforms(config.IMAGE_SIZE)
    valid_tfms = dataset.get_valid_transforms(config.IMAGE_SIZE)

    train_dataset = dataset.SegmentationDataset(
        train_images, train_masks, train_tfms,
        config.CLASS_COLORS, config.CLASSES, config.CLASSES
    )
    valid_dataset = dataset.SegmentationDataset(
        valid_images, valid_masks, valid_tfms,
        config.CLASS_COLORS, config.CLASSES, config.CLASSES
    )

    train_data_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)

    # Model, Optimizer, Loss Function
    cfz_model = model.CFZNet(classes=config.NUM_OF_CLASSES)
    cfz_model = utils.to_device(cfz_model)
    print(f"Model has {utils.get_model_parameters(cfz_model)} parameters.")

    optimizer = torch.optim.Adam(cfz_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Training
    NUM_EPOCHS = 25 # You can adjust the number of epochs
    train_model(
        cfz_model,
        train_data_loader,
        valid_data_loader,
        optimizer,
        loss_fn,
        NUM_EPOCHS,
        config.CLASSES,
        config.SAVE_PATH
    )

    # Optional: Load best model and visualize a batch
    # cfz_model.load_state_dict(torch.load(os.path.join(config.SAVE_PATH, "best_model.pth")))
    # cfz_model.eval()
    # (inputs, targets) = next(iter(valid_data_loader))
    # with torch.no_grad():
    #     outputs = cfz_model(utils.to_device(inputs))
    #     predictions = torch.argmax(outputs, dim=1)
    # visualize_batch(inputs, targets, predictions)