from .. import _base_model
import modules.SCR.layers as layers
import modules.SCR.losses as losses
import torch
import yaml
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any, Iterator
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from importlib import import_module
import time
from torch import Tensor


class SCRFull(_base_model._Model_Warpper):
    def __init__(self, conf_file, device):
        super(SCRFull, self).__init__(conf_file, device)
        ''' Get configuration '''
        with open(conf_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            args = config['model']['args']

        # Initialize encoder, decoder, hyper-encoder, and hyper-decoder layers
        self.encoder = layers.Encoder(**args)
        self.decoder = layers.Decoder(input_channel=3, **args)
        self.hyper_encoder = layers.HyperEncoder(**args)
        self.hyper_decoder = layers.HyperDecoderSC(**args)

        # Initialize entropy bottleneck with specified parameters
        self.entropy_bottleneck = EntropyBottleneck(args['N'], filters=args['filters'], tail_mass=2 ** -8)

        # Define scales for conditional bottleneck
        SCALES_MIN = args['scales_min']
        SCALES_MAX = args['scales_max']
        SCALES_LEVELS = args['scales_levels']
        scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)).tolist()

        # Initialize conditional bottleneck with scale table
        self.conditional_bottleneck = GaussianConditional(scale_table, tail_mass=2**-8)
        self.conditional_bottleneck.update()

        # Initialize learnable parameters (Quantization Vector QV/IQV,  importance adjustment factor gamma)
        self.qv_eps = 1e-5
        self.QV = torch.nn.Parameter(torch.ones((8, 320)))
        self.IQV = torch.nn.Parameter(torch.ones((8, 320)))
        self.gamma = torch.nn.Parameter(torch.ones((8, 320)))

        # Initialize loss functions with specified weights
        self.losses = {loss: getattr(losses, loss)(config['loss_w'][i]) for i, loss in enumerate(config['loss'])}

        # Update entropy bottleneck and conditional bottleneck
        self.entropy_bottleneck.update()
        self.conditional_bottleneck.update()
        self.to(device)  # Specify location

    def train(self, mode: bool = True):
        # Train
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.hyper_encoder.train(mode)
        self.hyper_decoder.train(mode)

        self.entropy_bottleneck.train(mode)
        self.conditional_bottleneck.train(mode)

        # Grad
        self.encoder.requires_grad_(mode)
        self.decoder.requires_grad_(mode)
        self.hyper_encoder.requires_grad_(mode)
        self.hyper_decoder.requires_grad_(mode)

        self.entropy_bottleneck.requires_grad_(mode)
        self.conditional_bottleneck.requires_grad_(mode)

        self.QV.requires_grad_(mode)
        self.IQV.requires_grad_(mode)
        self.gamma.requires_grad_(mode) if hasattr(self, 'gamma') else None  # for SCRWithoutSC

        # Status
        self.training = mode

    def param(self) -> Dict[str, Iterator[torch.nn.Parameter]]:
        # Return parameter groups for optimizer
        return {'main': self.parameters(), 'aux': self.entropy_bottleneck.parameters()}

    def forward(self, images: torch.Tensor, quality_level: Optional[Union[float, int]] = None) -> Dict[str, Any]:
        if self.training:  # Train
            return self.train_forward(images)
        else:  # Test
            return self.test_forward(images, quality_level)

    def train_forward(self, images: Tensor) -> Dict[str, Any]:
        x = images / 127.5 - 1.  # Normalize input images to the range [-1, 1]
        y = self.encoder(x)  # Encode input images
        z = self.hyper_encoder(y)  # Apply hyper-encoder
        # z_hat, _ = self.entropy_bottleneck(z, training=False)

        # Compress z with the entropy bottleneck (for training)
        z_tilde, z_likelihood = self.entropy_bottleneck(z, training=True)

        pred_sigma, imp_map = self.hyper_decoder(z_tilde)  # Decode z with the hyper-decoder
        # pred_sigma, imp_map = self.hyper_decoder(z_hat)

        # Initialize lists to store reconstructed images and likelihoods
        recon_x_list = []
        z_likelihoods_list = []
        y_likelihoods_list = []

        # Iterate over 8 quantization levels
        for q_lv in range(8):
            # Take Quantization vectors and importance adjustment factor for specific quantization level
            QV = torch.clamp(self.QV[q_lv][None, :, None, None], min=0) + self.qv_eps  # QV/IQV should be positive
            IQV = torch.clamp(self.IQV[q_lv][None, :, None, None], min=0) + self.qv_eps
            gamma = self.gamma[q_lv][None, :, None, None]

            adjusted_importance_map = torch.pow(imp_map, gamma)  # Importance adjustment

            # Create a relaxed mask for quantization
            samples = adjusted_importance_map + (torch.rand_like(adjusted_importance_map) - 0.5)
            mask_relaxed = samples + samples.round().detach() - samples.detach()  # Differentiable torch.round()

            # Compress y with the conditional bottleneck (for training)
            y_tilde, y_likelihood = self.conditional_bottleneck(y / QV, pred_sigma / QV, training=True)
            # y_hat, _ = self.conditional_bottleneck(y / QV, pred_sigma / QV, training=False)

            # Apply inverse quantization
            y_tilde = y_tilde * IQV
            # y_hat = y_hat * IQV

            # Combine quantized representations with the relaxed mask
            combined_y_tilde = mask_relaxed * y_tilde

            # Decode the combined representation
            x_recon = self.decoder(combined_y_tilde)
            recon_x_list.append(x_recon)
            y_likelihoods_list.append(y_likelihood)
            z_likelihoods_list.append(z_likelihood)

        # Concatenate reconstructed images and likelihoods
        recon_images = torch.clip((torch.cat(recon_x_list, dim=0) + 1) * 127.5, 0.0, 255.0)
        y_likelihoods = torch.cat(y_likelihoods_list, dim=0)
        z_likelihoods = torch.cat(z_likelihoods_list, dim=0)

        # Prepare arguments for loss calculation
        loss_args = {
            'images': images, 'recon_images': recon_images,
            'y_likelihoods': y_likelihoods, 'z_likelihoods': z_likelihoods,
        }

        # Calculate the total loss and auxiliary loss from the entropy bottleneck
        loss, loss_dict = self.loss(loss_args)
        aux_loss = self.entropy_bottleneck.loss()
        loss_dict['aux_loss'] = aux_loss

        # Create an output dictionary with loss components and reconstructed images
        output_dict = {'loss': loss, 'aux_loss': aux_loss, 'recon_images': recon_images, 'loss_dict': loss_dict}

        return output_dict

    def test_forward(self, images: torch.Tensor, quality_level: Union[float, int]) -> Dict[str, Any]:
        # Encode the input images
        encoder_output = self.encode(images, quality_level)

        # Decode the encoded strings
        decoder_output = self.decode(encoder_output['strings'], encoder_output['size'], quality_level)

        # Combine encoder and decoder output into a single dictionary
        output_dict = {**encoder_output, **decoder_output}

        return output_dict

    def encode(self, images: torch.Tensor, quality_level: Union[float, int]) -> Dict[str, Any]:
        # Normalize input images to the range [-1, 1]
        x = images / 127.5 - 1.

        # Encode input images
        y = self.encoder(x)
        z = self.hyper_encoder(y)

        # Compress 'z' representation
        z_hat, _ = self.entropy_bottleneck(z, training=False)
        z_string = self.entropy_bottleneck.compress(z)

        # Handle device migration for testing (To avoid ConvTranspose2D inconsistency in gpu)
        self.hyper_decoder.to('cpu')
        pred_sigma, imp_map = self.hyper_decoder(z_hat.to('cpu'))
        pred_sigma, imp_map = pred_sigma.to(x.device), imp_map.to(x.device)
        self.hyper_decoder.to(x.device)

        # Prepare scaling vectors and mask
        QV, IQV = self.get_scaling_vectors(quality_level)
        mask = self.get_mask(imp_map, quality_level)

        # Extract sigma and 'y' data for encoding with selective compression
        sigma_input = (pred_sigma / QV)[mask].unsqueeze(0)
        y_input = (y / QV)[mask].unsqueeze(0)

        # Build indexes and compress 'y'
        indexes = self.conditional_bottleneck.build_indexes(sigma_input)
        y_string = self.conditional_bottleneck.compress(y_input, indexes)

        # Create an output dictionary
        output_dict = {
            'strings': {'y': y_string, 'z': z_string},
            'quality_level': quality_level,
            'size': z.size()[-2:],
        }

        return output_dict

    def decode(self, string_dict: Dict[str, str], size: Tuple[int, int], quality_level: Union[float, int]) -> Dict[str, Any]:
        # Obtain scaling vectors
        QV, IQV = self.get_scaling_vectors(quality_level)

        # Extract 'y' and 'z' strings from the dictionary
        y_string = string_dict['y']
        z_string = string_dict['z']

        # Decompress 'z' representation
        z_hat = self.entropy_bottleneck.decompress(z_string, size)

        # Handle device migration for testing
        self.hyper_decoder.to('cpu')
        pred_sigma, imp_map = self.hyper_decoder(z_hat.to('cpu'))
        pred_sigma, imp_map = pred_sigma.to(QV.device), imp_map.to(QV.device)
        self.hyper_decoder.to(QV.device)

        # Create a mask based on importance map
        mask = self.get_mask(imp_map, quality_level)

        # Scale 'pred_sigma' and extract sigma values based on the mask
        pred_sigma = pred_sigma / QV
        sigma_input = self.selection_from_mask(pred_sigma, mask)

        # Build indexes and decompress 'y' strings
        indexes = self.conditional_bottleneck.build_indexes(sigma_input)
        y_hat_flat = self.conditional_bottleneck.decompress(y_string, indexes)

        # Combine 'y' and scale it with 'IQV'
        combined_y_hat = torch.zeros_like(pred_sigma)
        combined_y_hat[mask] = y_hat_flat[0]
        combined_y_hat = combined_y_hat * IQV

        # Decode the combined representation to obtain reconstructed images
        recon_x = self.decoder(combined_y_hat)

        # Clip and round the reconstructed images
        recon_images = torch.round(torch.clip((recon_x + 1) * 127.5, 0.0, 255.0))

        # Create an output dictionary
        output_dict = {'recon_images': recon_images}

        return output_dict

    def get_scaling_vectors(self, quality_level: Union[float, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaling vectors (QV and IQV) based on the specified quality level.

        Args:
            quality_level (int or float): The quality level used to compute scaling vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing QV (Quality Vector) and IQV (Inverse Quality Vector).

        This method calculates scaling vectors QV and IQV based on the provided quality level.
        If the quality level is a float, it retrieves the corresponding scaling vectors from pre-defined values (QVs).
        If the quality level is not an integer, it performs linear interpolation between the scaling
        vectors of the two nearest integer quality levels.

        Example usage:
        ```
        quality_level = 3  # Integer quality level
        QV, IQV = self.get_scaling_vectors(quality_level)

        quality_level = 2.5  # Non-integer quality level
        QV, IQV = self.get_scaling_vectors(quality_level)
        ```
        """
        if quality_level - (quality_level // 1) == 0:
            # If quality_level is an integer, retrieve scaling vectors for the exact quality level.
            quality_level = int(quality_level)
            idx = quality_level - 1

            QV = torch.clamp(self.QV[idx][None, :, None, None], min=0) + self.qv_eps
            IQV = torch.clamp(self.IQV[idx][None, :, None, None], min=0) + self.qv_eps
        else:
            # If quality_level is not an integer, perform linear interpolation between two nearest integer levels.
            low_quality_level = int(quality_level)
            high_quality_level = int(low_quality_level + 1)
            decimal_part = quality_level - low_quality_level

            low_QV = torch.clamp(self.QV[low_quality_level - 1][None, :, None, None], min=0) + self.qv_eps
            low_IQV = torch.clamp(self.IQV[low_quality_level - 1][None, :, None, None], min=0) + self.qv_eps

            high_QV = torch.clamp(self.QV[high_quality_level - 1][None, :, None, None], min=0) + self.qv_eps
            high_IQV = torch.clamp(self.IQV[high_quality_level - 1][None, :, None, None], min=0) + self.qv_eps

            # Perform linear interpolation to compute QV and IQV.
            QV = low_QV ** (1 - decimal_part) * high_QV ** decimal_part
            IQV = low_IQV ** (1 - decimal_part) * high_IQV ** decimal_part

        return QV, IQV

    def get_mask(self, importance_map: torch.Tensor, quality_level: Union[float, int]) -> torch.Tensor:
        """
        Compute a mask based on the importance map and quality level.

        Args:
            importance_map (torch.Tensor): The importance map used to compute the mask.
            quality_level (Union[float, int]): The quality level for which the mask is computed.

        Returns:
            torch.Tensor: A binary mask indicating the selected regions based on the importance map.

        This method computes a binary mask based on the provided importance map and quality level. If the quality level
        is an integer, it calculates the mask directly by raising the importance map to the power of the corresponding
        gamma value and rounding it. If the quality level is not an integer, it performs linear interpolation between
        gamma values of the two nearest integer quality levels to calculate the mask.

        Example usage:
        ```
        importance_map = torch.rand((batch_size, channels, height, width))
        quality_level = 3  # Integer quality level
        mask = self.get_mask(importance_map, quality_level)

        quality_level = 2.5  # Non-integer quality level
        mask = self.get_mask(importance_map, quality_level)
        ```
        """
        if quality_level - (quality_level // 1) == 0:
            # If quality_level is an integer, retrieve the corresponding gamma value and calculate the mask.
            quality_level = int(quality_level)
            idx = quality_level - 1

            gamma = self.gamma[idx][None, :, None, None]
            adjusted_importance_map = torch.pow(importance_map, gamma)
            mask = torch.round(adjusted_importance_map)

        else:
            # If quality_level is not an integer, perform linear interpolation between gamma values.
            low_quality_level = int(quality_level)
            high_quality_level = int(low_quality_level + 1)
            decimal_part = quality_level - low_quality_level

            gamma_low = self.gamma[low_quality_level - 1][None, :, None, None]
            gamma_high = self.gamma[high_quality_level - 1][None, :, None, None]

            gamma = gamma_low ** (1 - decimal_part) * gamma_high ** decimal_part

            adjusted_importance_map = torch.pow(importance_map, gamma)
            mask = torch.round(adjusted_importance_map)

        return mask.to(torch.bool)

    @staticmethod
    def selection_from_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Select elements from the tensor based on the provided mask.

        Args:
            tensor (torch.Tensor): The input tensor from which elements are selected.
            mask (torch.Tensor): The binary mask indicating which elements to select.

        Returns:
            torch.Tensor: A tensor containing the selected elements.
        """
        mask_flat = mask.ravel()
        idx = torch.nonzero(mask_flat, as_tuple=True)
        tensor_flat = tensor.ravel()
        reshaped_tensor = tensor_flat[idx]

        return reshaped_tensor.unsqueeze(0)

    def load(self, model_dir):
        state_dict = torch.load(model_dir)  # Read state_dict from saved file
        self.load_state_dict(state_dict, strict=False)  # Load weight
        self.entropy_bottleneck.update(True)  # Update buffer (_quantized_cdf, _offset, _cdf_length)
        self.conditional_bottleneck.update()  # Update buffer (_quantized_cdf, _offset, _cdf_length, scale_table)
        # self.conditional_bottleneck.update_scale_table(scale_table, True)  # if scale_table changes


class SCRWithoutSC(SCRFull):
    def __init__(self, conf_file, device):
        super().__init__(conf_file, device)
        ''' Get configuration '''
        with open(conf_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            args = config['model']['args']
        self.hyper_decoder = layers.HyperDecoder(**args)
        self.hyper_decoder.to(self.device)
        del self.gamma  # without selective compression

    def train_forward(self, images: Tensor) -> Dict[str, Any]:
        x = images / 127.5 - 1.  # Normalize input images to the range [-1, 1]
        y = self.encoder(x)  # Encode input images
        z = self.hyper_encoder(y)  # Apply hyper-encoder
        # z_hat, _ = self.entropy_bottleneck(z, training=False)

        # Compress z with the entropy bottleneck (for training)
        z_tilde, z_likelihood = self.entropy_bottleneck(z, training=True)

        pred_sigma = self.hyper_decoder(z_tilde)  # Decode z with the hyper-decoder
        # pred_sigma, imp_map = self.hyper_decoder(z_hat)

        # Initialize lists to store reconstructed images and likelihoods
        recon_x_list = []
        z_likelihoods_list = []
        y_likelihoods_list = []

        # Iterate over 8 quantization levels
        for q_lv in range(8):
            # Take Quantization vectors and importance adjustment factor for specific quantization level
            QV = torch.clamp(self.QV[q_lv][None, :, None, None], min=0) + self.qv_eps  # QV/IQV should be positive
            IQV = torch.clamp(self.IQV[q_lv][None, :, None, None], min=0) + self.qv_eps

            # Compress y with the conditional bottleneck (for training)
            y_tilde, y_likelihood = self.conditional_bottleneck(y / QV, pred_sigma / QV, training=True)
            # y_hat, _ = self.conditional_bottleneck(y / QV, pred_sigma / QV, training=False)

            # Apply inverse quantization
            y_tilde = y_tilde * IQV
            # y_hat = y_hat * IQV

            # Decode the combined representation
            x_recon = self.decoder(y_tilde)
            recon_x_list.append(x_recon)
            y_likelihoods_list.append(y_likelihood)
            z_likelihoods_list.append(z_likelihood)

        # Concatenate reconstructed images and likelihoods
        recon_images = torch.clip((torch.cat(recon_x_list, dim=0) + 1) * 127.5, 0.0, 255.0)
        y_likelihoods = torch.cat(y_likelihoods_list, dim=0)
        z_likelihoods = torch.cat(z_likelihoods_list, dim=0)

        # Prepare arguments for loss calculation
        loss_args = {
            'images': images, 'recon_images': recon_images,
            'y_likelihoods': y_likelihoods, 'z_likelihoods': z_likelihoods,
        }

        # Calculate the total loss and auxiliary loss from the entropy bottleneck
        loss, loss_dict = self.loss(loss_args)
        aux_loss = self.entropy_bottleneck.loss()
        loss_dict['aux_loss'] = aux_loss

        # Create an output dictionary with loss components and reconstructed images
        output_dict = {'loss': loss, 'aux_loss': aux_loss, 'recon_images': recon_images, 'loss_dict': loss_dict}

        return output_dict

    def encode(self, images: torch.Tensor, quality_level: Union[float, int]) -> Dict[str, Any]:
        # Normalize input images to the range [-1, 1]
        x = images / 127.5 - 1.

        # Encode input images
        y = self.encoder(x)
        z = self.hyper_encoder(y)

        # Compress 'z' representation
        z_hat, _ = self.entropy_bottleneck(z, training=False)
        z_string = self.entropy_bottleneck.compress(z)

        # Handle device migration for testing (To avoid ConvTranspose2D inconsistency in gpu)
        self.hyper_decoder.to('cpu')
        pred_sigma = self.hyper_decoder(z_hat.to('cpu'))
        pred_sigma = pred_sigma.to(x.device)
        self.hyper_decoder.to(x.device)

        # Prepare scaling vectors and mask
        QV, IQV = self.get_scaling_vectors(quality_level)

        # Extract sigma and 'y' data for encoding
        sigma_input = pred_sigma / QV
        y_input = y / QV

        # Build indexes and compress 'y'
        indexes = self.conditional_bottleneck.build_indexes(sigma_input)
        y_string = self.conditional_bottleneck.compress(y_input, indexes)

        # Create an output dictionary
        output_dict = {
            'strings': {'y': y_string, 'z': z_string},
            'quality_level': quality_level,
            'size': z.size()[-2:],
        }

        return output_dict

    def decode(self, string_dict: Dict[str, str], size: Tuple[int, int], quality_level: Union[float, int]) -> Dict[str, Any]:
        # Obtain scaling vectors
        QV, IQV = self.get_scaling_vectors(quality_level)

        # Extract 'y' and 'z' strings from the dictionary
        y_string = string_dict['y']
        z_string = string_dict['z']

        # Decompress 'z' representation
        z_hat = self.entropy_bottleneck.decompress(z_string, size)

        # Handle device migration for testing
        self.hyper_decoder.to('cpu')
        pred_sigma = self.hyper_decoder(z_hat.to('cpu'))
        pred_sigma = pred_sigma.to(QV.device)
        self.hyper_decoder.to(QV.device)

        # Scale 'pred_sigma' and extract sigma values based on the mask
        pred_sigma = pred_sigma / QV

        # Build indexes and decompress 'y' strings
        indexes = self.conditional_bottleneck.build_indexes(pred_sigma)
        y_hat = self.conditional_bottleneck.decompress(y_string, indexes)
        y_hat = y_hat * IQV

        # Decode the combined representation to obtain reconstructed images
        recon_x = self.decoder(y_hat)

        # Clip and round the reconstructed images
        recon_images = torch.round(torch.clip((recon_x + 1) * 127.5, 0.0, 255.0))
        recon_images = recon_images.to(QV.device)

        # Create an output dictionary
        output_dict = {'recon_images': recon_images}

        return output_dict

    def get_mask(self, **kwargs):
        raise Exception("This function is deprecated and should not be used in 'without selective compression' mode.")

    def selection_from_mask(self, **kwargs):
        raise Exception("This function is deprecated and should not be used in 'without selective compression' mode.")