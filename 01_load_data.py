#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:30:06 2026

@author: livboethrysoe
"""


from pathlib import Path
from torchvision import datasets, transforms

"""
Script for downloading the CIFAR-10 and CIFAR-100 data sets.
"""

# Set the dataset directory relative to this script
data_dir = Path(__file__).resolve().parent / "data"

# Convert images to PyTorch tensors
transform = transforms.ToTensor()

# Load CIFAR-10
cifar10_train = datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

cifar10_test = datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=True,
    transform=transform
)

# Load CIFAR-100
cifar100_train = datasets.CIFAR100(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

cifar100_test = datasets.CIFAR100(
    root=data_dir,
    train=False,
    download=True,
    transform=transform
)

# Inspect the datasets
print("CIFAR-10 training images:", len(cifar10_train))
print("CIFAR-10 test images:", len(cifar10_test))
print("CIFAR-100 training images:", len(cifar100_train))
print("CIFAR-100 test images:", len(cifar100_test))
print("Image shape:", cifar10_train[0][0].shape)