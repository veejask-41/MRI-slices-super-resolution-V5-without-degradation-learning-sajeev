import torch
import os
import numpy as np
import nibabel as nib

from .SRUNet import SRUNet
from .degradation_network import DegradationNetwork
from .VGGStylePatchGAN import VGGStylePatchGAN

from utils.losses import perceptual_quality_loss, GDNLoss, perceptual_adversarial_loss


class SuperResolutionModel:
    def __init__(self, opt):
        # Initialize the models based on the configuration
        self.sr_unet = SRUNet(
            image_size=opt.image_size,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            freeze_encoder=opt.freeze_encoder,
        )
        self.degradation_network = DegradationNetwork(image_size=opt.image_size)
        self.vgg_patch_gan = VGGStylePatchGAN(patch_size=70)

        # Optimizers for SRUNet and VGGStylePatchGAN only, since DegradationNetwork is not trained
        self.optimizer_sr = torch.optim.Adam(self.sr_unet.parameters(), lr=opt.lr)
        self.optimizer_gan = torch.optim.Adam(
            self.vgg_patch_gan.parameters(), lr=opt.lr
        )

        # Placeholder for storing input data
        self.lr_slices = []
        self.hr_slices = []

        self.current_losses = {}
        self.current_visuals = {}

    def set_input(self, data):
        """
        Prepares the input data by extracting slices from a 3D MRI volume.

        Args:
            data (dict): A dictionary containing 'LR' and 'HR' keys with respective 3D volumes.
                        - data['LR']: Low-resolution MRI volume (shape: [150, 256, 256])
                        - data['HR']: High-resolution MRI volume (shape: [150, 256, 256])
        """
        # Clear previous slice data
        self.lr_slices.clear()
        self.hr_slices.clear()

        # Clear visuals for new data
        self.current_visuals["SR"] = []

        # Ensure data has the right structure and load LR and HR volumes
        lr_volume = data["LR"]  # Expected shape: (150, 256, 256)
        hr_volume = data["HR"]  # Expected shape: (150, 256, 256)

        # Check the dimensions of the input data
        assert (
            lr_volume.shape == hr_volume.shape
        ), "LR and HR volumes must have the same shape"
        num_slices, height, width = lr_volume.shape

        # Process each slice individually
        for i in range(num_slices):
            # Extract each slice and add a channel dimension (assumed to be grayscale images)
            lr_slice = lr_volume[i][None, :, :]  # Shape: (1, 256, 256)
            hr_slice = hr_volume[i][None, :, :]  # Shape: (1, 256, 256)

            # Convert slices to tensors and move to the appropriate device
            lr_tensor = torch.tensor(lr_slice, dtype=torch.float32).unsqueeze(
                0
            )  # Shape: (1, 1, 256, 256)
            hr_tensor = torch.tensor(hr_slice, dtype=torch.float32).unsqueeze(
                0
            )  # Shape: (1, 1, 256, 256)

            # Move tensors to the device set for the model
            lr_tensor = lr_tensor.to(self.sr_unet.device)
            hr_tensor = hr_tensor.to(self.sr_unet.device)

            # Append to the slice list
            self.lr_slices.append(lr_tensor)
            self.hr_slices.append(hr_tensor)

    def get_slice_pair(self, index):
        """
        Retrieves the LR and HR tensors for a specific slice index.

        Args:
            index (int): The index of the slice to retrieve.

        Returns:
            tuple: A tuple (lr_slice, hr_slice) containing the LR and HR slice tensors.
        """
        if index < 0 or index >= len(self.lr_slices):
            raise IndexError("Slice index out of range")

        return self.lr_slices[index], self.hr_slices[index]

    def optimize_parameters(self, lr_images, hr_images, lambda_tv):
        # Step 1: Forward pass through SRUNet
        sr_output = self.sr_unet(lr_images)

        # Store SR output for later visualization
        if "SR" not in self.current_visuals:
            self.current_visuals["SR"] = []
        self.current_visuals["SR"].append(sr_output.squeeze().cpu().numpy())

        # Step 2: Pass SRUNet output through DegradationNetwork to compute the feedback loss
        blur_kernel = self.degradation_network(sr_output)

        # Step 3: Prepare input for VGGStylePatchGAN

        # Forward pass through VGGStylePatchGAN
        real_pred = self.vgg_patch_gan(hr_images)
        fake_pred = self.vgg_patch_gan(sr_output)

        # Step 4: Calculate losses
        # Loss for SRUNet
        loss_sr = perceptual_quality_loss(
            sr_output, hr_images, alpha=1.0, beta=1.0, gamma=1.0
        )

        # Feedback loss from DegradationNetwork - affects SRUNet
        loss_gdn = GDNLoss(sr_output, lr_images, blur_kernel, lambda_tv)

        # Loss for VGGStylePatchGAN
        loss_gan = perceptual_adversarial_loss(
            hr_images,
            sr_output,
            real_pred,
            fake_pred,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
            delta=1.0,
        )

        # Total loss for SRUNet includes both perceptual and degradation feedback losses
        total_loss_sr = loss_sr + loss_gdn

        # Step 5: Backpropagation and optimization for SRUNet
        self.optimizer_sr.zero_grad()
        total_loss_sr.backward(
            retain_graph=True
        )  # Retain graph for subsequent backpropagation
        self.optimizer_sr.step()

        # Backpropagation and optimization for VGGStylePatchGAN
        self.optimizer_gan.zero_grad()
        loss_gan.backward()
        self.optimizer_gan.step()

        # Calculate losses
        loss_results = {
            "loss_sr": loss_sr.item(),
            "loss_gdn": loss_gdn.item(),
            "loss_gan": loss_gan.item(),
        }

        # Store the current losses
        self.current_losses = loss_results

    def get_current_losses(self):
        """
        Returns the losses for the currently processed slice.

        Returns:
            dict: A dictionary containing the current losses: 'loss_sr', 'loss_gdn', and 'loss_gan'.
        """

        # Return the most recently stored losses
        return self.current_losses

    def compute_visuals(self):
        if "SR" not in self.current_visuals or len(self.current_visuals["SR"]) != len(
            self.lr_slices
        ):
            print(
                "SR visuals are not fully computed; make sure to run optimize_parameters for all slices."
            )

    def save_volume(self, epoch):
        """
        Assembles and saves the current LR and SR slices into 3D MRI volumes in NIfTI format.

        Args:
            epoch (int): The current epoch number for folder naming and file naming.
        """
        # Create folder for saving the volumes
        save_folder = os.path.join("results", f"epoch_{epoch}")
        os.makedirs(save_folder, exist_ok=True)

        # Assemble LR and SR slices into 3D volumes
        lr_volume = np.stack(
            [lr_slice.squeeze().cpu().numpy() for lr_slice in self.lr_slices], axis=0
        )
        sr_volume = np.stack(
            [self.current_visuals["SR"].squeeze() for _ in range(len(self.lr_slices))],
            axis=0,
        )

        # Save LR and SR volumes as .nii files
        lr_img = nib.Nifti1Image(lr_volume, affine=np.eye(4))
        sr_img = nib.Nifti1Image(sr_volume, affine=np.eye(4))

        lr_file_path = os.path.join(save_folder, f"epoch{epoch}_LR.nii")
        sr_file_path = os.path.join(save_folder, f"epoch{epoch}_HR.nii")

        nib.save(lr_img, lr_file_path)
        nib.save(sr_img, sr_file_path)

        print(f"Saved LR volume to {lr_file_path}")
        print(f"Saved SR volume to {sr_file_path}")

    def save_final_models(self):
        """
        Saves the final trained SRUNet and VGGStylePatchGAN models to disk after training completion.
        """
        # Specify the folder and file paths for saving the models
        save_folder = "final_models"
        os.makedirs(save_folder, exist_ok=True)
        sr_unet_file_path = os.path.join(save_folder, "SRUNet_final.pth")
        vgg_patch_gan_file_path = os.path.join(
            save_folder, "VGGStylePatchGAN_final.pth"
        )

        # Save the models
        torch.save(self.sr_unet.state_dict(), sr_unet_file_path)
        torch.save(self.vgg_patch_gan.state_dict(), vgg_patch_gan_file_path)

        print(f"Saved final SRUNet model to {sr_unet_file_path}")
        print(f"Saved final VGGStylePatchGAN model to {vgg_patch_gan_file_path}")
