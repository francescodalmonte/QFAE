import os
import random

from PIL import Image
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors


def is_image(filename):
    exts = ['.jpg', '.jpeg', '.png', '.bmp', 
            '.JPG', '.JPEG', '.PNG', '.BMP']
    return any(filename.endswith(extension) for extension in exts)


class CustomBMADDataset(Dataset):
    """Base class for BMAD custom datasets."""

    def __init__(self, dataset_path, transforms, seed=None):
        self.dataset_path = dataset_path
        self.seed = seed

        if transforms is not None:
            self.transform = transforms
        
        # load dataset
        self.x_paths, self.mask_paths, self.labels = self.load_dataset_folder()

    def __getitem__(self, idx):
        x_path, mask_path, label = self.x_paths[idx], self.mask_paths[idx], self.labels[idx]
        x = Image.open(x_path).convert('RGB')
        if mask_path is None:
            mask = torch.zeros([1, x.size[0], x.size[1]])
            y = torch.as_tensor(0)
        else:
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
            else:
                mask = torch.zeros([1, x.size[0], x.size[1]])
            y = torch.as_tensor(label)
        # apply transformations
        x, mask = tv_tensors.Image(x), tv_tensors.Mask(mask)
        if self.transform is not None:
            x, mask = self.transform(x, mask)

        return x, y, mask, x_path

    def __len__(self):
        return len(self.x_paths)

    def load_dataset_folder(self):
        raise NotImplementedError("This is a parent class!")



class CustomBMADTestDataset(CustomBMADDataset):
    """Custom dataset for testing.
    dataset_path:   path to the dataset folder (expected structure: dataset_path/good/img, dataset_path/Ungood/img, dataset_path/Ungood/anomaly_mask).
    transforms:     custom transformations to be applied to the images and masks.
    max_samples:    maximum number of samples to load from the dataset.

    """

    def __init__(self, dataset_path, transforms=None, max_samples=None, seed=None):
        self.max_samples = max_samples
        super().__init__(dataset_path, transforms, seed)

    def load_dataset_folder(self):
        """Loop through the provided dataset folder and return lists of
        all filenames for images and masks."""

        x_paths = []
        mask_paths = []
        labels = []

        dirpath_good = os.path.join(self.dataset_path, "good/img")
        dirpath_anomalous = os.path.join(self.dataset_path, "Ungood/img")
        dirpath_masks_good = os.path.join(self.dataset_path, "good/label")
        dirpath_masks_anomalous = os.path.join(self.dataset_path, "Ungood/label")

        for file in os.listdir(dirpath_good):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                x_paths.append(os.path.join(dirpath_good, file))
                mask_paths.append(os.path.join(dirpath_masks_good, file))
                labels.append(0)
        for file in os.listdir(dirpath_anomalous):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                x_paths.append(os.path.join(dirpath_anomalous, file))
                mask_paths.append(os.path.join(dirpath_masks_anomalous, file))
                labels.append(1)


        # shuffle the dataset
        zipped = list(zip(x_paths, mask_paths, labels))
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(zipped)
        x_paths, mask_paths, labels = zip(*zipped)

        # subset if max_samples is provided
        if self.max_samples is not None:
            x_paths = x_paths[:self.max_samples]
            mask_paths = mask_paths[:self.max_samples]
            labels = labels[:self.max_samples]

        return x_paths, mask_paths, labels
    



class CustomBMADTrainDataset_oneclass(CustomBMADDataset):
    """Custom dataset for training (anomaly detection/one-class).
    dataset_path:   path to the dataset folder (expected structure: dataset_path/good/img).
    transforms:     custom transformations to be applied to the images and masks.
    max_samples:    maximum number of samples to load from the dataset.
    
    """

    def __init__(self, dataset_path, transforms=None, max_samples=None, seed=None):
        self.max_samples = max_samples
        super().__init__(dataset_path, transforms, seed)

    def load_dataset_folder(self):
        """Loop through the provided dataset folder and return lists of
        filenames for images and masks to be used for training."""

        x_paths = []
        mask_paths = []
        labels = []

        dirpath_good = os.path.join(self.dataset_path, "good/img")

        for file in os.listdir(dirpath_good):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                x_paths.append(os.path.join(dirpath_good, file))
                mask_paths.append(None)
                labels.append(0)

        # shuffle the dataset
        zipped = list(zip(x_paths, mask_paths, labels))
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(zipped)
        x_paths, mask_paths, labels = zip(*zipped)

        # subset if max_samples is provided
        if self.max_samples is not None:
            x_paths = x_paths[:self.max_samples]
            mask_paths = mask_paths[:self.max_samples]
            labels = labels[:self.max_samples]

        return x_paths, mask_paths, labels


class ImagenetteDataset(Dataset):
    def __init__(self, data_path, transforms=None, seed=42,
                 max_samples=np.inf, prefetch=False):
        self.data_path = data_path
        self.transforms = transforms
        self.seed = seed
        self.max_samples = max_samples
        self.prefetch = prefetch

        # load dataset
        self.x = self.load_dataset_folder()

        # prefetch data if needed
        if self.prefetch:
            self.prefetch_data()

    def get_from_path(self, x_path):
        # load image 
        x = Image.open(x_path).convert('RGB')
        x = tv_tensors.Image(x)

        # apply transforms
        if self.transforms is not None:
            x = self.transforms(x)

        return x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.prefetch:
            return x
        else:
            return self.get_from_path(x)

    def load_dataset_folder(self):
        """Loop through the dataset folder and load images paths."""

        x = []

        root = os.path.join(self.data_path, "train")
        img_folders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        for folder in img_folders:
            for file in os.listdir(folder):
                if is_image(file):
                    x.append(os.path.join(folder, file))

        # shuffle the dataset
        if self.seed is not None:
            np.random.seed(self.seed)
        np.random.shuffle(x)

        # clip the dataset if needed
        if self.max_samples < len(x):
            x = x[:self.max_samples]
        
        return x
    
    def prefetch_data(self):
        print("Prefetching data...")
        for idx in range(len(self)):
            print(f"{idx}/{len(self)}", end="\r")
            self.x[idx] = self.get_from_path(self.x[idx])
    

class InfiniteDataLoader:
    """borrowed from https://discuss.pytorch.org/t/infinite-dataloader/17903/16"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)  # Reset the data loader
            data = next(self.data_iter)
        return data


def custom_collate_fn(batch):
    x, y, mask, p = zip(*batch)
    x, mask = torch.stack(x, dim=0), torch.stack(mask, dim=0)
    y = torch.tensor(y)
    return x, y, mask, p
