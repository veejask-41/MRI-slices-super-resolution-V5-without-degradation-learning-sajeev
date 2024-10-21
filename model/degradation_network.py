import torch
import torch.nn as nn
import torch.fft
import torchvision.transforms.functional as TF
from PIL import Image
import math
import torch.nn.functional as F


class DegradationNetwork(nn.Module):
    def __init__(self, image_size, slice_thickness_ratio=0.5, fwhm=1.0):
        super(DegradationNetwork, self).__init__()
        self.image_size = image_size
        self.slice_thickness_ratio = slice_thickness_ratio
        self.fwhm = fwhm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.low_pass_filter = self.create_sinc_gaussian_filter()
        print(f"Low-pass filter shape: {self.low_pass_filter.shape}")  # Debug

    def forward(self, x, angle, translation):
        print(f"Input shape: {x.shape}")  # Debug
        transformed_img_tensor = self.apply_rigid_transform(x, angle, translation)
        print(f"Transformed image shape: {transformed_img_tensor.shape}")  # Debug

        freq_domain = torch.fft.fftshift(torch.fft.fft2(transformed_img_tensor))
        print(f"Frequency domain shape: {freq_domain.shape}")  # Debug

        filtered_freq = freq_domain * self.low_pass_filter
        print(f"Filtered frequency shape: {filtered_freq.shape}")  # Debug

        degraded_image = torch.fft.ifft2(torch.fft.ifftshift(filtered_freq)).real
        print(f"Degraded image shape: {degraded_image.shape}")  # Debug
        return degraded_image

    def create_sinc_gaussian_filter(self):
        x = torch.linspace(-1, 1, self.image_size, device=self.device)
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        radius = torch.sqrt(xx**2 + yy**2)

        sinc_filter = torch.where(
            radius == 0,
            torch.ones_like(radius),
            torch.sin(radius / self.slice_thickness_ratio)
            / (radius / self.slice_thickness_ratio),
        )
        gaussian_filter = torch.exp(-0.5 * (radius / self.fwhm) ** 2)
        combined_filter = sinc_filter * gaussian_filter
        return combined_filter

    def apply_rigid_transform(self, image, angle, translation):
        if not torch.is_tensor(image):
            image = TF.to_tensor(image)
        image = image.unsqueeze(0)  # Adding batch dimension for processing
        print(
            f"Image tensor shape after adding batch dimension: {image.shape}"
        )  # Debug

        angle_rad = torch.tensor(angle * (math.pi / 180), device=self.device)
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
        tx, ty = translation

        affine_matrix = torch.tensor(
            [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], device=self.device
        ).unsqueeze(
            0
        )  # Batch dimension for affine_grid
        print(f"Affine matrix: {affine_matrix}")  # Debug

        grid = F.affine_grid(affine_matrix, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)
        print(
            f"Transformed image shape after grid_sample: {transformed_image.shape}"
        )  # Debug

        return transformed_image.squeeze(0)


# Example use
# img_size = 256
# input_image = torch.rand(
#     1,
#     img_size,
#     img_size,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
# )

# degradation_net = DegradationNetwork(image_size=img_size)
# angle = 30  # Rotation angle in degrees
# translation = (0.1, 0.1)  # Translation parameters
# degraded_image = degradation_net(input_image, angle, translation)
# print(f"Final degraded image shape: {degraded_image.shape}")
