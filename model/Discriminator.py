import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from GDN import GDNModel


# Multi-GDN Model with Multiple Degradation Networks for Different Patterns
class MultiGDNModel(nn.Module):
    def __init__(self, gdn_params, in_channels=1, out_channels=1):
        super(MultiGDNModel, self).__init__()

        # Create multiple GDNs with unique degradation parameters
        self.gdns = nn.ModuleList(
            [
                GDNModel(
                    size=params["size"],
                    sigma=params["sigma"],
                    sinc_scale=params["sinc_scale"],
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
                for params in gdn_params
            ]
        )

    def forward(self, x):
        # Collect outputs from each GDN separately
        outputs = [gdn(x) for gdn in self.gdns]
        return outputs  # List of outputs from each GDN


# Define specific parameters for each GDN
gdn_params = [
    {"size": 9, "sigma": 1.0, "sinc_scale": 0.3},  # GDN 1
    {"size": 11, "sigma": 1.5, "sinc_scale": 0.5},  # GDN 2
    {"size": 13, "sigma": 2.0, "sinc_scale": 0.7},  # GDN 3
]

# Instantiate the Multi-GDN Model
model = MultiGDNModel(gdn_params, in_channels=1, out_channels=1)

# Example Loss Functions for Each GDN (can be any appropriate function)
loss_functions = [
    nn.MSELoss(),  # Loss for GDN 1
    nn.L1Loss(),  # Loss for GDN 2
    nn.SmoothL1Loss(),  # Loss for GDN 3
]

# Example Target Outputs (must match your actual data and GDN count)
targets = [
    torch.randn(1, 1, 256, 256),  # Target for GDN 1
    torch.randn(1, 1, 256, 256),  # Target for GDN 2
    torch.randn(1, 1, 256, 256),
]  # Target for GDN 3

# Sample input for testing
x = torch.randn(1, 1, 256, 256)  # Example input with batch size 1 and single channel

# Forward pass through Multi-GDN model
outputs = model(x)

# Calculate individual losses for each GDN's output
total_loss = 0.0
for output, target, loss_fn in zip(outputs, targets, loss_functions):
    loss = loss_fn(output, target)
    total_loss += loss  # Accumulate the total loss

# Print the total loss
print("Total Loss:", total_loss.item())
print(len(outputs))
