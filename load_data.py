#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:30:06 2026

@author: livboethrysoe
"""


from pathlib import Path
from torchvision import datasets, transforms

"""
Load CIFAR-10 and CIFAR-100 datasets.

If the datasets are not already present in the data directory,
they will be downloaded automatically.
"""

# Dataset directory
DATA_DIR = Path(__file__).resolve().parent / "data"

# Convert images to PyTorch tensors
transform = transforms.ToTensor()


def load_cifar10():
    """
    Load CIFAR-10 training and test datasets.
    """

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def load_cifar100():
    """
    Load CIFAR-100 training and test datasets.
    """

    train_dataset = datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR100(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


if __name__ == "__main__":

    # CIFAR-10
    cifar10_train, cifar10_test = load_cifar10()

    print("=== CIFAR-10 ===")
    print(f"Training images: {len(cifar10_train)}")
    print(f"Test images: {len(cifar10_test)}")
    print(f"Image shape: {cifar10_train[0][0].shape}")

    print()

    # CIFAR-100
    cifar100_train, cifar100_test = load_cifar100()

    print("=== CIFAR-100 ===")
    print(f"Training images: {len(cifar100_train)}")
    print(f"Test images: {len(cifar100_test)}")
    print(f"Image shape: {cifar100_train[0][0].shape}")