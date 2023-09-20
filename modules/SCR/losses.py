import torch
import torch.nn as nn
import numpy as np


class Distortion(nn.Module):
    def __init__(self, weight: float):
        """
        Initializes the Distortion loss used in the NeurIPS'22
        "Selective compression learning of latent representations for variable-rate image compression" paper.

        Args:
            weight (float): Weighting factor for the loss.
        """
        super().__init__()
        self.weight = weight
        lambda_list = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]  # 8 scales for 8 quantization level
        self.scales_tensor = torch.tensor(lambda_list).view(-1, 1, 1, 1)  # (8, 1, 1, 1)

    def forward(self, images: torch.Tensor, recon_images: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the Distortion loss module.

        Args:
            images (torch.Tensor): Original images. (B, C, H, W)
            recon_images (torch.Tensor): Reconstructed images. (B, C, H, W)
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Computed loss value.
        """
        # Check the batch sizes of images and recon_images
        batch_size_images, _, _, _ = images.size()
        batch_size_recon, _, _, _ = recon_images.size()

        # If the batch sizes of images and recon_images are different, adjust the batch size
        if batch_size_images != batch_size_recon:
            # Copy images to match the batch size of recon_images
            rate = batch_size_recon // batch_size_images  # rate = 8
            extend_images = torch.cat([images] * rate, dim=0)

        # Calculate the MSE loss
        scales = self.scales_tensor.repeat((batch_size_images, 1, 1, 1)).to(images.device)  # (batch_size * 8, 1, 1, 1)
        loss = ((scales * (extend_images - recon_images)) ** 2).mean() * self.weight

        return loss


class Rate(nn.Module):
    def __init__(self, weight: float):
        """
        Initializes the Rate loss used in the NeurIPS'22
        "Selective compression learning of latent representations for variable-rate image compression" paper.

        Args:
            weight (float): Weighting factor for the loss.
        """
        super().__init__()
        self.weight = weight

    def forward(self, images: torch.Tensor, y_likelihoods: torch.Tensor, z_likelihoods: torch.Tensor, **kwargs):
        """
        Forward pass of the Rate loss module.

        Args:
            images (torch.Tensor): Original images. (B, C, H, W); For calculating num_pixels
            y_likelihoods (torch.Tensor): Likelihood of y from entropy bottleneck
            z_likelihoods (torch.Tensor): Likelihood of z from GaussianCondition
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Computed loss value.
        """
        num_pixels = images.shape[-2] * images.shape[-1]
        B = y_likelihoods.shape[0]
        log_y_likelihoods = torch.log(y_likelihoods)
        log_z_likelihoods = torch.log(z_likelihoods)

        estimated_y_bpp = torch.sum(log_y_likelihoods) / (-np.log(2) * num_pixels * B)
        estimated_z_bpp = torch.sum(log_z_likelihoods) / (-np.log(2) * num_pixels * B)

        loss = estimated_y_bpp + estimated_z_bpp * self.weight

        return loss
