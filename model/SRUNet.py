import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# from .networks import GeneratorFrequencyFilter


# class SRUNet(nn.Module):
#     def __init__(self, image_size, in_channels=1, out_channels=1, freeze_encoder=True):
#         super(SRUNet, self).__init__()

#         # Initialize frequency filter
#         self.frequency_filter = GeneratorFrequencyFilter(image_size)

#         # Initialize UNet with EfficientNet-b3 encoder
#         self.unet = smp.Unet(
#             encoder_name="efficientnet-b3",  # Suitable for MRI super-resolution
#             encoder_weights="imagenet",  # Pre-trained weights
#             in_channels=in_channels,  # Typically 1 for grayscale MRI images
#             classes=out_channels,  # Number of output channels
#         )

#         # Optionally freeze encoder layers to prevent updating
#         if freeze_encoder:
#             for param in self.unet.encoder.parameters():
#                 param.requires_grad = False

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x_filtered = self.frequency_filter(x)
#         x_unet = self.unet(x_filtered)
#         return self.sigmoid(x_unet)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Convolutional layer with padding to keep the spatial dimensions unchanged
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)  # Apply convolution
        x = self.sigmoid(x)  # Apply sigmoid activation
        return x


# # Create the network
# net = SimpleNet()

# # Example input tensor
# input_tensor = torch.randn(1, 1, 256, 256)  # Random tensor simulating an input

# # Forward pass
# output_tensor = net(input_tensor)

# print(output_tensor.shape)  # Should print torch.Size([1, 1, 256, 256])


# Define image size and instantiate model
# image_size = 256
# model = SRUNet(
#     image_size=image_size, in_channels=1, out_channels=1, freeze_encoder=True
# )

# model = SRUNet(n_channels=1, n_classes=1)

# # Input image
# x = torch.randn(
#     1, 1, image_size, image_size
# )  # Example input with batch size 1, grayscale

# # Forward pass
# output = model(x)
# print(output.shape)
