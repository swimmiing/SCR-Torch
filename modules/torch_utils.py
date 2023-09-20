import torch
import random
import os
import numpy as np
import torchvision.transforms as transforms
from typing import Union, Sequence
import collections


def fix_seed(seed: int = 0):
    """
    Fix random seeds for reproducibility.

    Args:
        seed (int): Seed value for random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id: int):
    """
    Set random seed for PyTorch DataLoader workers, including Distributed Data Parallel (DDP) setups.

    Args:
        worker_id (int): Worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_img_from_tensor(tensor_data: torch.Tensor, directory: str, names: Sequence[str]):
    """
    Save images from tensor data to a specified directory.

    Args:
        tensor_data (torch.Tensor): 4D tensor containing image data (batch_size, channels, height, width).
        directory (str): Directory where images will be saved.
        names (Sequence[str]): List of image names corresponding to the batch.
    """
    os.makedirs(directory, exist_ok=True)
    for i in range(tensor_data.shape[0]):
        tensor_image = tensor_data[i].byte()
        pil_image = transforms.ToPILImage()(tensor_image)
        pil_image.save(os.path.join(directory, names[i] + '.png'))

