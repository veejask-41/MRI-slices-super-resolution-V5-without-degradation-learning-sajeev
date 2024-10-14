import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from networks import GDNGaussianSincFilter


# GDN Model with pre-trained UNet
class GDNModel(nn.Module):
    def __init__(self, size, sigma, sinc_scale, in_channels=1, out_channels=1):
        super(GDNModel, self).__init__()

        # Load a pre-trained UNet model
        self.unet = smp.Unet(
            encoder_name="resnet34",  # or "efficientnet-b0", etc.
            encoder_weights="imagenet",  # Use "imagenet" weights
            in_channels=in_channels,  # Set to 1 for grayscale images
            classes=out_channels,
        )

        # Freeze encoder layers if desired
        for param in self.unet.encoder.parameters():
            param.requires_grad = False

        # Gaussian-Sinc Filter for frequency domain processing
        self.filter = GDNGaussianSincFilter(size, sigma, sinc_scale)

    def forward(self, x):
        # Use the pre-trained UNet to estimate the degradation
        x = self.unet(x)

        # Apply the Gaussian-Sinc filter in the frequency domain
        x = self.filter(x)

        return x


# Instantiate the model with pre-trained UNet for grayscale input
model = GDNModel(size=11, sigma=1.5, sinc_scale=0.5, in_channels=1, out_channels=1)

# Sample input for testing (single-channel grayscale)
x = torch.randn(1, 1, 256, 256)  # Example input with batch size 1 and single channel
output = model(x)
print("Final output shape:", output.shape)
