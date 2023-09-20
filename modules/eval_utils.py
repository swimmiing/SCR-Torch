import torch
import os
import numpy as np
from typing import Sequence, Dict, Union
from compressai.utils.bench.codecs import _compute_psnr, _compute_ms_ssim


class ImgQEvaluator(object):
    def __init__(self):
        """
        Image Quality Evaluator class for evaluating image compression metrics.

        Initializes PSNR, BPP (bits per pixel), and MS-SSIM metrics.
        """
        super(ImgQEvaluator, self).__init__()
        self.psnr = []
        self.bpp = []
        self.ms_ssim = []
        metrics = ['PSNR', 'BPP', 'MS-SSIM']
        self.metric_dict = {k: 0.0 for k in metrics}

    def evaluate_batch(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            strings: Sequence[bytes],
            num_pixels: int
    ) -> Dict[str, float]:
        """
        Evaluate image quality metrics for a batch of predictions.

        Args:
            pred (torch.Tensor): Predicted image data as a tensor.
            target (torch.Tensor): Target image data as a tensor.
            strings (Sequence[bytes]): List of compressed strings for the batch.
            num_pixels (int): Number of pixels in the images.

        Returns:
            Dict[str, float]: A dictionary containing PSNR, BPP, and MS-SSIM metrics.
        """
        psnr = self.cal_PSNR(pred, target)
        bpp = self.cal_BPP(strings, num_pixels)
        ms_ssim = self.cal_MultiScaleSSIM(pred, target)
        return {'PSNR': psnr, 'BPP': bpp, 'MS-SSIM': ms_ssim}

    def cal_PSNR(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) for a batch of images.

        Args:
            pred (torch.Tensor): Predicted image data as a tensor.
            target (torch.Tensor): Target image data as a tensor.

        Returns:
            float: The PSNR value.
        """
        psnr = _compute_psnr(target, pred)
        self.psnr.append(psnr)
        return psnr

    def cal_BPP(self, strings: Sequence[bytes], num_pixels: int) -> float:
        """
        Calculate Bits Per Pixel (BPP) for a batch of compressed strings.

        Args:
            strings (Sequence[bytes]): List of compressed strings for the batch.
            num_pixels (int): Number of pixels in the images.

        Returns:
            float: The BPP value.
        """
        bpp = sum(len(item) for string in strings for item in string) * 8 / num_pixels
        self.bpp.append(bpp)
        return bpp


    def cal_MultiScaleSSIM(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate Multi-Scale Structural Similarity Index (MS-SSIM) for a batch of images.

        Args:
            pred (torch.Tensor): Predicted image data as a tensor.
            target (torch.Tensor): Target image data as a tensor.

        Returns:
            float: The MS-SSIM value.
        """
        ms_ssim = _compute_ms_ssim(target, pred)
        self.ms_ssim.append(ms_ssim)
        return ms_ssim

    def finalize_PSNR(self) -> float:
        """
        Calculate the final PSNR metric by averaging over all batches.

        Returns:
            float: The averaged PSNR value.
        """
        psnr = np.mean(self.psnr)
        self.metric_dict['PSNR'] = psnr
        return psnr

    def finalize_BPP(self) -> float:
        """
        Calculate the final BPP metric by averaging over all batches.

        Returns:
            float: The averaged BPP value.
        """
        bpp = np.mean(self.bpp)
        self.metric_dict['BPP'] = bpp
        return bpp

    def finalize_MultiScaleSSIM(self) -> float:
        """
        Calculate the final MS-SSIM metric by averaging over all batches.

        Returns:
            float: The averaged MS-SSIM value.
        """
        ms_ssim = np.mean(self.ms_ssim)
        self.metric_dict['MS-SSIM'] = ms_ssim
        return ms_ssim

    def finalize(self) -> Dict[str, float]:
        """
        Calculate and finalize all image quality metrics.

        Returns:
            Dict[str, float]: A dictionary containing the final PSNR, BPP, and MS-SSIM metrics.
        """
        self.finalize_PSNR()
        self.finalize_BPP()
        self.finalize_MultiScaleSSIM()
        return self.metric_dict


def save_string_dict(string_dict: Dict[str, bytes], directory: str, names: Sequence[str]):
    """
    Save compressed strings to a specified directory.

    Args:
        string_dict (Dict[str, bytes]): A dictionary of compressed strings.
        directory (str): Directory where compressed strings will be saved.
        names (Sequence[str]): List of image names corresponding to the batch.
    """
    for i, name in enumerate(names):
        os.makedirs(os.path.join(directory, name), exist_ok=True)
        for k, strings in string_dict.items():
            with open(os.path.join(directory, name, f'{k}.bin'), "wb") as file:
                file.write(strings[i])


def write_metric_log(
        model_name: str,
        metric_dict: Dict[str, float],
        log_path: str,
        msg_postfix: str = "",
        mode: str = 'w') -> str:
    """
    Write image compression metrics to a log file.

    Args:
        model_name (str): The name of the image compression model.
        metric_dict (Dict[str, float]): A dictionary containing quality metrics.
        log_path (str): The path to the log file.
        msg_postfix (str, optional): An optional postfix for the log message (default: "").
        mode (str): mode is an optional string that specifies the mode in which the file
    is opened.
    Returns:
        msg (str): Log message
    """
    # Construct the log message
    msg = f'Compression result {msg_postfix}\n'
    msg += f'model: {model_name}\n'
    msg += "".join([f"{k}: {v:.7f};  " for k, v in metric_dict.items()]) + "\n"

    # Write messages to the log file
    with open(log_path, mode) as fp_rst:
        fp_rst.write(msg)

    return msg
