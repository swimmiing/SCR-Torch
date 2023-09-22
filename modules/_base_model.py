from abc import ABC, abstractmethod
import torch
from torch import nn
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.models.utils import update_registered_buffers
from typing import Any, Mapping, Dict, Tuple, Iterator
# import yaml


class _Model_Warpper(nn.Module):
    def __init__(self, conf_file: str, device: str):
        """
        Initialize the _Model instance.

        Args:
            conf_file (str): Path to the configuration file.
            device (str): Device to run the model on (e.g., 'cuda' or 'cpu').

        This constructor loads the model configuration from a YAML file and sets the device.
        """
        super(_Model_Warpper, self).__init__()

        ''' Get configuration '''
        # with open(conf_file) as f:
        #     config = yaml.load(f, Loader=yaml.FullLoader)

        self.device = device

    def train(self, **kwargs):
        """
        Placeholder for the training logic.

        Override this method to specify which layers to freeze or fine-tune during training.
        """
        raise NotImplementedError()

    def param(self) -> Dict[str, Iterator[torch.nn.Parameter]]:
        """
        Placeholder for the parameter grouping logic for optimizer.

        Override this method to group parameters for each optimizer during training.
        """

    def encode(self, **kwargs) -> Dict[str, Any]:
        """
        Placeholder for the encoding logic.

        Override this method to define the encoder's input-output structure.
        """
        raise NotImplementedError()

    def decode(self, **kwargs) -> Dict[str, Any]:
        """
        Placeholder for the decoding logic.

        Override this method to define the decoder's input-output structure.
        """
        raise NotImplementedError()

    def loss(self, loss_args: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate the total loss and individual loss components (excluding auxiliary losses).

        Args:
            loss_args (Dict[str, Any]): A dictionary containing the necessary arguments for loss computation.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing the total loss and a dictionary
            of individual loss components.

        This method calculates the total loss by summing the individual loss components specified in `self.losses`.
        The `self.losses` dictionary should be defined with loss module instances, and this method computes
        each loss component based on the provided `loss_args`. The total loss is the sum of all individual losses,
        excluding auxiliary losses.

        Example usage:
        ```
        loss_args = {
            'images': input_images,
            'recon_images': reconstructed_images,
            'y_likelihoods': y_loss,
            'z_likelihoods': z_loss,
        }

        self.losses = {
            'Distortion': torch.nn.Module,
            'Rate': torch.nn.Module,
        }
        ```
        """
        loss_dict = {}
        for loss_name, loss_module in self.losses.items():
            # Calculate each individual loss component and store it in the loss_dict.
            loss_dict[loss_name] = loss_module(**loss_args)

        # Sum all individual losses to compute the total loss.
        total_loss = torch.sum(torch.stack(list(loss_dict.values())))

        return total_loss, loss_dict

    def forward(self, **kwargs) -> Dict[str, Any]:
        """
        Placeholder for the forward pass logic.

        Override this method to define the forward pass of your model.
        """
        raise NotImplementedError()

    def train_forward(self, **kwargs) -> Dict[str, Any]:
        """
        Placeholder for the train forward pass logic.

        Override this method to define the train forward pass of your model.

        Returns:
            Dict[str, Any]: A dictionary containing relevant training outputs.
        Example:
            {
                'loss': torch.tensor(0.123),  # Primary loss value (Sum of loss_components except aux_loss)
                'aux_loss': torch.tensor(0.045),  # Auxiliary loss value (optional)
                'recon_images': torch.Tensor([...]),  # Reconstructed images
                'loss_dict': {
                    'loss_component1': torch.tensor(0.01),
                    'loss_component2': torch.tensor(0.02),
                    ...
                    'aux_loss': torch.tensor(0.045)
                }  # Additional loss components (optional)
            }
        """
        raise NotImplementedError()

    def test_forward(self, **kwargs) -> Dict[str, Any]:
        """
        Placeholder for the test forward pass logic.

        Override this method to define the test forward pass of your model.

        Returns:
            Dict[str, Any]: A dictionary containing relevant training outputs.
        Example:
            {
                'recon_images': torch.Tensor([...]),  # Reconstructed images
                'strings': {
                    'y': string list of y representation for batch
                    'z': string list of z representation for batch
                }
                'quality_level': int,
                ...
            }
        """
        raise NotImplementedError()

    def save(self, directory: str):
        torch.save(self.state_dict(), directory)

    def load(self, **kwargs):
        """
        Placeholder for the model loading logic.

        Override this method to specify how to load a saved model.
        """
        raise NotImplementedError()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        """
        Load the model's state dictionary with additional handling for specific modules.

        Args:
            state_dict (Dict[str, Any]): A dictionary containing the model's state.
            strict (bool): Whether to strictly enforce that the keys in `state_dict` match the model's structure.

        This method extends the base `load_state_dict` method to handle specific modules within the model.
        The added handling addresses the issue related to Entropy Model buffers not pre-allocated with sufficient space.

        The additional code checks for specific modules (e.g., EntropyBottleneck and GaussianConditional) and updates
        their registered buffers based on the provided `state_dict`. This update is necessary to handle issues related
        to buffer resizing.

        Returns:
            Module: The loaded model.
        """
        # Iterate through all named modules in the model.
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            # Update registered buffers for the EntropyBottleneck module
            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                    'resize'
                )

            # Update registered buffers for the GaussianConditional module
            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                    'resize'
                )
        return super().load_state_dict(state_dict, strict)
