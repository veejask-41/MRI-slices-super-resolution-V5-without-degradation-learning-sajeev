import torch
import torch.nn as nn
import torchvision.models as models
import torch.fft


class CustomMiniPatchGAN(nn.Module):
    def __init__(self):
        super(CustomMiniPatchGAN, self).__init__()

        # Initial convolution layer to reduce the channels from 512 to 256
        self.initial = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        # Intermediate layers including downsampling and normalization
        self.middle = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                256, 128, kernel_size=3, stride=2, padding=1
            ),  # Downsampling to 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1
            ),  # Maintaining size 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final convolution layer to get to 1 output channel
        self.final = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.initial(x)  # Output: [1, 256, 8, 8]
        x = self.middle(x)
        x = self.final(x)
        return x


class SingleChannelVGG(nn.Module):
    def __init__(self):
        super(SingleChannelVGG, self).__init__()
        vgg19 = models.vgg19(pretrained=True)

        # Load original VGG features
        original_vgg_features = vgg19.features

        # Modify the first convolutional layer
        # New layer with 1 input channel, but the same output channels and kernel size
        first_conv_layer = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # Copy the weights from the first three channels of the pretrained model
        first_conv_layer.weight.data = (
            original_vgg_features[0].weight.data[:, 0:1, :, :].clone()
        )
        first_conv_layer.bias.data = original_vgg_features[0].bias.data.clone()

        # Replace the first conv layer in the features list
        modified_features = nn.Sequential(
            first_conv_layer, *list(original_vgg_features.children())[1:]
        )

        # Assign modified features to the class variable
        self.vgg_layers = modified_features

    def forward(self, x):
        x = self.vgg_layers(x)
        return x


class FrequencyFiltering(nn.Module):
    def __init__(self, image_size):
        super(FrequencyFiltering, self).__init__()
        # Create a Gaussian-sinc combined filter in frequency domain
        self.register_buffer("filter", self.create_gaussian_sinc_filter(image_size))

    def forward(self, x):
        # Apply FFT
        x_fft = torch.fft.fft2(x)
        # Filter in the frequency domain
        filtered_fft = x_fft * self.filter
        # Apply Inverse FFT
        x_filtered = torch.fft.ifft2(
            filtered_fft
        ).real  # Use the real part for image data
        return x_filtered

    def create_gaussian_sinc_filter(self, size):
        # filter: Modify as per actual sinc and Gaussian parameters
        sigma = size / 6.0  # Standard deviation for Gaussian
        x = torch.arange(size).float() - size / 2
        x, y = torch.meshgrid(x, x)
        r = torch.sqrt(x**2 + y**2)
        sinc = torch.sin(r) / r
        sinc[r == 0] = 1  # Handling NaN at the center
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return (sinc * gaussian).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)
