import torch
import torch.nn as nn
from compressai.layers import GDN
from typing import List, Optional, Union, Tuple
from compressai.models.utils import conv, deconv

''' Type Hint '''
_size_2_t = Union[int, Tuple[int, int]]  # int or tuple


class Conv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1):
        """
        Initializes a 2D convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple, optional): Stride for the convolution operation. Defaults to 1.

        Note:
            This convolutional layer is designed to provide behavior similar to TensorFlow's "padding='same'"
            option for 'Conv2D', where the layer automatically calculates padding values to ensure the output
            has the same spatial dimensions as the input.
        """
        super(Conv2D, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # Zero-padding for PyTorch Conv2d
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the convolution operation.
        """
        # Measure input size
        input_height, input_width = x.size(2), x.size(3)
        output_height, output_width = x.size(2) // self.conv.stride[0], x.size(3) // self.conv.stride[1]

        # Calculate the required padding values
        pad_height = max((output_height - 1) * self.conv.stride[0] + self.conv.kernel_size[0] - input_height, 0)
        pad_width = max((output_width - 1) * self.conv.stride[1] + self.conv.kernel_size[1] - input_width, 0)

        # Consider stride when calculating padding
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding and perform Convolution operation
        x = nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        x = self.conv(x)
        return x


class Deconv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1):
        """
        Initializes a 2D transposed convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the transposed convolutional kernel.
            stride (int or tuple, optional): Stride for the transposed convolution operation. Defaults to 1.

        Note:
            This transposed convolutional layer is designed to provide behavior similar to TensorFlow's
            "padding='same'" option for 'Conv2DTranspose', where the layer automatically calculates padding values
            to ensure the output has the same spatial dimensions as the input.
        """
        super(Deconv2D, self).__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transposed convolutional layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the transposed convolution operation.
        """
        # Measure input size
        input_height, input_width = x.size(2), x.size(3)
        output_height, output_width = x.size(2) * self.deconv.stride[0], x.size(3) * self.deconv.stride[1]

        # Calculate the required padding values
        pad_height = max((input_height - 1) * self.deconv.stride[0] + self.deconv.kernel_size[0] - output_height, 0)
        pad_width = max((input_width - 1) * self.deconv.stride[1] + self.deconv.kernel_size[1] - output_width, 0)

        # Consider stride when calculating padding
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding and perform transposed convolution operation
        x = self.deconv(x)
        x = x[:, :, pad_left:-pad_right, pad_top:-pad_bottom]

        return x


class Encoder(nn.Module):
    def __init__(self, N, M, **kwargs):
        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            Conv2D(in_channels=3, out_channels=N, kernel_size=5, stride=2),
            GDN(N)
        )
        self.conv2 = nn.Sequential(
            Conv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
            GDN(N)
        )
        self.conv3 = nn.Sequential(
            Conv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
            GDN(N)
        )

        self.conv4 = nn.Sequential(
            Conv2D(in_channels=N, out_channels=M, kernel_size=5, stride=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_channel, N, M, **kwargs):
        super(Decoder, self).__init__()

        self.deconv1 = nn.Sequential(
            Deconv2D(in_channels=M, out_channels=N, kernel_size=5, stride=2),
            GDN(N, inverse=True)
        )
        self.deconv2 = nn.Sequential(
            Deconv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
            GDN(N, inverse=True)
        )
        self.deconv3 = nn.Sequential(
            Deconv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
            GDN(N, inverse=True)
        )
        self.deconv4 = nn.Sequential(
            Deconv2D(in_channels=N, out_channels=input_channel, kernel_size=5, stride=2)
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x


class HyperEncoder(nn.Module):
    def __init__(self, N, M, **kwargs):
        super(HyperEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            Conv2D(in_channels=M, out_channels=N, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            Conv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            Conv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
        )

    def forward(self, x):
        x = torch.abs(x)
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out


class HyperDecoderSC(nn.Module):
    def __init__(self, N, M, **kwargs):
        super(HyperDecoderSC, self).__init__()

        self.deconv1 = nn.Sequential(
            Deconv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            Deconv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            Deconv2D(in_channels=N, out_channels=M, kernel_size=3, stride=1),
        )

        self.relu = nn.ReLU()

        self.mask_conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=N, out_channels=M, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        penultimate = x
        x = self.deconv3(x)
        sigma = self.relu(x)
        out2 = self.mask_conv(penultimate)
        importance_map = torch.clip(out2 + 0.5, 0, 1)
        return sigma, importance_map


class HyperDecoder(nn.Module):
    def __init__(self, N, M, **kwargs):
        super(HyperDecoder, self).__init__()

        self.deconv1 = nn.Sequential(
            Deconv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            Deconv2D(in_channels=N, out_channels=N, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            Deconv2D(in_channels=N, out_channels=M, kernel_size=3, stride=1),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        sigma = self.relu(x)
        return sigma
