
import torch.nn as nn
import segmentation_models_pytorch as smp
import config

class CFZNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=config.NUM_OF_CLASSES):
        super(CFZNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )

    def forward(self, x):
        return self.model(x)