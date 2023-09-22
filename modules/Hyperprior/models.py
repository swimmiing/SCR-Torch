from .. import _base_model
import modules.SCR.layers as layers
import modules.Hyperprior.losses as losses
import torch
import yaml
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any, Iterator
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from importlib import import_module
import time
from torch import Tensor


class Hyperprior(_base_model._Model_Warpper):
    def __init__(self, conf_file, device):
        super(Hyperprior, self).__init__(conf_file, device)
        ''' Get configuration '''
        with open(conf_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            args = config['model']['args']
            args['N'] = 192 if args['quality_level'] in [6, 7, 8] else 128
            args['M'] = 320 if args['quality_level'] in [6, 7, 8] else 192
            lambda_list = [0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
            config['loss_w'][0] = lambda_list[int(args['quality_level'])-1]  # Distortion loss weight

        # Initialize encoder, decoder, hyper-encoder, and hyper-decoder layers
        self.encoder = layers.Encoder(**args)
        self.decoder = layers.Decoder(input_channel=3, **args)
        self.hyper_encoder = layers.HyperEncoder(**args)
        self.hyper_decoder = layers.HyperDecoder(**args)

        # Initialize entropy bottleneck with specified parameters
        self.entropy_bottleneck = EntropyBottleneck(args['N'], filters=args['filters'], tail_mass=2 ** -8)

        # Define scales for conditional bottleneck
        SCALES_MIN = 0.11
        SCALES_MAX = 256
        SCALES_LEVELS = 64
        scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)).tolist()

        # Initialize conditional bottleneck with scale table
        self.conditional_bottleneck = GaussianConditional(scale_table, tail_mass=2**-8)
        self.conditional_bottleneck.update()

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
        z_tilde, z_likelihoods = self.entropy_bottleneck(z, training=True)

        pred_sigma, imp_map = self.hyper_decoder(z_tilde)  # Decode z with the hyper-decoder
        # pred_sigma, imp_map = self.hyper_decoder(z_hat)

        # Compress y with the conditional bottleneck (for training)
        y_tilde, y_likelihoods = self.conditional_bottleneck(y. pred_sigma, training=True)
        # y_hat, _ = self.conditional_bottleneck(y / QV, pred_sigma / QV, training=False)

        # Decode the combined representation
        x_recon = self.decoder(y_tilde)

        # Concatenate reconstructed images and likelihoods
        recon_images = torch.clip((x_recon + 1) * 127.5, 0.0, 255.0)

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
        pred_sigma = self.hyper_decoder(z_hat.to('cpu'))
        pred_sigma = pred_sigma.to(x.device)
        self.hyper_decoder.to(x.device)

        # Build indexes and compress 'y'
        y_hat, _ = self.conditional_bottleneck(y, pred_sigma, training=False)
        indexes = self.conditional_bottleneck.build_indexes(pred_sigma)
        y_string = self.conditional_bottleneck.compress(y_hat, indexes)

        # Create an output dictionary
        output_dict = {
            'strings': {'y': y_string, 'z': z_string},
            'quality_level': quality_level,
            'size': z.size()[-2:],
        }

        return output_dict

    def decode(self, string_dict: Dict[str, str], size: Tuple[int, int], quality_level: Union[float, int]) -> Dict[str, Any]:
        # Extract 'y' and 'z' strings from the dictionary
        y_string = string_dict['y']
        z_string = string_dict['z']

        # Decompress 'z' representation
        z_hat = self.entropy_bottleneck.decompress(z_string, size)

        # Handle device migration for testing
        self.hyper_decoder.to('cpu')
        pred_sigma = self.hyper_decoder(z_hat.to('cpu'))
        pred_sigma = pred_sigma.to(self.device)
        self.hyper_decoder.to(self.device)

        # Build indexes and decompress 'y' strings
        indexes = self.conditional_bottleneck.build_indexes(pred_sigma)
        y_hat = self.conditional_bottleneck.decompress(y_string, indexes)

        # Decode the combined representation to obtain reconstructed images
        recon_x = self.decoder(y_hat)

        # Clip and round the reconstructed images
        recon_images = torch.round(torch.clip((recon_x + 1) * 127.5, 0.0, 255.0))

        # Create an output dictionary
        output_dict = {'recon_images': recon_images}

        return output_dict

    def load(self, model_dir):
        state_dict = torch.load(model_dir)  # Read state_dict from saved file
        self.load_state_dict(state_dict, strict=False)  # Load weight
        self.entropy_bottleneck.update(True)  # Update buffer (_quantized_cdf, _offset, _cdf_length)
        self.conditional_bottleneck.update()  # Update buffer (_quantized_cdf, _offset, _cdf_length, scale_table)
        # self.conditional_bottleneck.update_scale_table(scale_table, True)  # if scale_table changes
