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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SRUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SRUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(768, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        self.up3 = Up(192, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        out = self.sigmoid(logits)

        return out


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
