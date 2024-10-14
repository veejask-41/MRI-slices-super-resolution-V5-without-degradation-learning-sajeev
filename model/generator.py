import torch
import torch.nn as nn
from networks import FrequencyFiltering
from RRDB_architecture import RRDBNet


class Generator(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, image_size=256):
        super(Generator, self).__init__()
        self.freq_filter = FrequencyFiltering(image_size)
        self.rrdb_net = RRDBNet(in_nc, out_nc, nf, nb, gc)

    def forward(self, x):
        x = self.freq_filter(x)  # Apply frequency domain filtering
        x = self.rrdb_net(x)  # Apply RRDB network for super-resolution
        return x


image_size = 256  # Define the image size used for the filter
in_nc = 1  # Number of input channels (e.g., RGB)
out_nc = 1  # Number of output channels (should match input if same space)
nf = 64  # Number of features
nb = 23  # Number of blocks

# Create generator
generator = Generator(in_nc, out_nc, nf, nb, image_size=image_size)

# Dummy input - replace with actual data
input_image = torch.randn(1, in_nc, image_size, image_size)  # Random noise image

# Get the output
output_image = generator(input_image)
print(output_image.shape)
