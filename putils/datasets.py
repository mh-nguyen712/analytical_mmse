import torch
from torchvision import datasets
from torch.utils.data import Dataset, Subset
import os
import torchvision.transforms.v2 as v2
import getpass


def get_dataset(
    name: str = "",
    root: str = None,
    train: bool = True,
    transform: v2.Compose = None,
    color: bool = True,
) -> Dataset:
    if name in datasets.__all__:
        return get_torch_dataset(name, root, train, transform)
    else:
        if not os.path.exists(root):
            raise ValueError(
                f"Dataset {name} not found at {root}. Please check the path."
            )
        dataset = ImageFolder(root, transform, color=color)
        size = len(dataset)
        if train:
            return Subset(
                dataset, range(min(size, 50000))
            )  # Use first 50k samples for training
        else:
            return Subset(dataset, range(min(size, 50000), size))

# Get a dataset from torchvision
def get_torch_dataset(
    name: str, root: str = None, train: bool = True, transform: v2.Compose = None
) -> Dataset:
    if transform is None:
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    assert name in datasets.__all__, f"Dataset {name} not found in torchvision.datasets"
    dataset = datasets.__dict__[name](
        root=root, train=train, transform=transform, download=True
    )
    return dataset


from PIL import Image
import torchvision.transforms.v2 as v2


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, color: bool = True):
        self.root = root
        if transform is None:
            transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
            )
        self.transform = transform
        self.color = color
        self.image_files = [
            f for f in os.listdir(root) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.image_paths = sorted([os.path.join(root, f) for f in self.image_files])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if not self.color:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


import numpy as np
import math

class TensorDataset:
    """
    A simple dataset class to wrap one or more input tensors (NumPy arrays).
    It mimics the torch.utils.data.Dataset interface.
    """
    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("Must provide at least one tensor.")
        
        self.tensors = tensors
        self.num_samples = len(tensors[0])
        
        # Check if all tensors have the same length
        for t in self.tensors:
            if len(t) != self.num_samples:
                raise ValueError("All tensors must have the same length along dimension 0.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # If there's only one tensor, return just that item, otherwise return a tuple.
        if len(self.tensors) == 1:
            return self.tensors[0][idx]
        return tuple(t[idx] for t in self.tensors)


class TensorDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Initializes the DataLoader.

        Args:
            dataset (TensorDataset): The dataset object to load data from.
            batch_size (int): The number of samples per batch.
            shuffle (bool): Whether to shuffle the data indices before each epoch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        # Calculate the number of batches required
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """
        Returns the iterator object (self) and performs shuffling if requested.
        """
        # 1. Determine the order of indices for the current epoch
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # 2. Reset the current batch index
        self.current_batch_idx = 0
        return self

    def __next__(self):
        """
        Retrieves the next batch of data.
        """
        if self.current_batch_idx >= self.num_batches:
            # Stop iteration when all batches are processed
            raise StopIteration
        
        start_index = self.current_batch_idx * self.batch_size
        end_index = min((self.current_batch_idx + 1) * self.batch_size, self.num_samples)
        
        batch_global_indices = self.indices[start_index:end_index]

        batch_data = self.dataset[batch_global_indices]
        
        self.current_batch_idx += 1
        
        return batch_data
    