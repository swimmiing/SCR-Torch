import torch
import torch.nn as nn
import numpy as np


class Distortion(nn.Module):
    def __init__(self, weight: float):
        """
        Initializes the Distortion loss

        Args:
            weight (float): Weighting factor for the loss.
        """
        super().__init__()
        self.weight = weight

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
        # Calculate the MSE loss
        loss = ((images - recon_images) ** 2).mean() * self.weight

        return loss


class Rate(nn.Module):
    def __init__(self, weight: float):
        """
        Initializes the Rate loss

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
