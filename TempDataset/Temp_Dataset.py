import torch
from torch.utils.data import Dataset

import numpy as np
import torchaudio
from torchvision import transforms as vt
from PIL import Image, ImageFilter
import os
from tqdm import tqdm
import csv
import json
from typing import Dict, Any, Optional


class TempDataset(Dataset):
    def __init__(self, data_path: str, split: str, is_train: bool = True, input_resolution: Optional[int] = None):
        """
        Initialize TempDataset.

        Args:
            data_path (str): Path to the dataset directory.
            split (str): The dataset split (e.g., 'train', 'val', 'test').
            is_train (bool): Whether the dataset is used for training (default: True).
            input_resolution (int): Input image resolution (default: 256; None for test).

        The dataset directory should contain image files in PNG format and metadata CSV files.
        """
        super(TempDataset, self).__init__()

        self.split = split
        self.csv_dir = os.path.join('TempDataset', 'metadata', f'{split}.csv')

        self.image_path = data_path
        image_files = set([fn.split('.')[0] for fn in os.listdir(self.image_path) if fn.endswith('.png')])

        subset = set([item[0].split('.')[0] for item in csv.reader(open(self.csv_dir))])
        self.file_list = list(image_files.intersection(subset))
        self.file_list = sorted(self.file_list)

        if is_train:
            input_resolution = 256 if input_resolution is None else input_resolution
            self.image_transform = vt.Compose([
                vt.ToTensor(),
                vt.Lambda(lambda img: img * 255.0),  # [0, 255]
                vt.RandomCrop((input_resolution, input_resolution), pad_if_needed=True),
                vt.RandomHorizontalFlip(),
            ])
        else:
            self.image_transform = vt.Compose([
                vt.ToTensor(),
                vt.Lambda(lambda img: img * 255.0),  # [0, 255]
            ])

        self.is_train = is_train

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.
        """
        return len(self.file_list)

    def get_image(self, item: int) -> Image.Image:
        """
        Load and return an image from the dataset.

        Args:
            item (int): Index of the item to retrieve.

        Returns:
            Image.Image: Loaded image.
        """
        image_file = Image.open(os.path.join(self.image_path, f'{self.file_list[item]}.png'))
        return image_file

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Retrieve an item from the dataset.

        Args:
            item (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the image and its associated ID.
        """
        file_id = self.file_list[item]

        image_file = self.get_image(item)
        image = self.image_transform(image_file)

        out = {'images': image, 'ids': file_id}
        out = {key: value for key, value in out.items() if value is not None}
        return out
