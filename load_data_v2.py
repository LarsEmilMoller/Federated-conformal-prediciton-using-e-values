#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:53:58 2026

@author: livboethrysoe
"""


from pathlib import Path

from torchvision import datasets, transforms


"""
Load CIFAR-10 and CIFAR-100 datasets.

Training data uses data augmentation.
Calibration and test data use only tensor conversion and normalization.
"""


# --------------------------------------------------
# Dataset directory
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"


# --------------------------------------------------
# Normalization statistics
# --------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


# --------------------------------------------------
# CIFAR transforms
# --------------------------------------------------

def get_transforms(dataset_name):
    """
    Return training and evaluation transforms for a dataset.

    Training uses data augmentation.
    Evaluation uses only tensor conversion and normalization.
    """

    if dataset_name == "CIFAR10":
        mean = CIFAR10_MEAN
        std = CIFAR10_STD

    elif dataset_name == "CIFAR100":
        mean = CIFAR100_MEAN
        std = CIFAR100_STD

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            size=32,
            padding=4,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])

    return train_transform, eval_transform


# --------------------------------------------------
# CIFAR-10
# --------------------------------------------------

def load_cifar10():
    """
    Load CIFAR-10 training and test datasets.

    The training dataset uses data augmentation.
    The test dataset uses only normalization.
    """

    train_transform, eval_transform = get_transforms(
        "CIFAR10"
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=eval_transform,
    )

    return train_dataset, test_dataset


def load_cifar10_calibration():
    """
    Load the CIFAR-10 training images without augmentation.

    This dataset contains the same underlying training images
    in the same order as the dataset returned by load_cifar10().
    It allows prepare_data.py to select calibration examples
    without applying random training augmentations.

    This is not the official CIFAR-10 test dataset.
    """

    _, eval_transform = get_transforms("CIFAR10")

    calibration_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=eval_transform,
    )

    return calibration_dataset


# --------------------------------------------------
# CIFAR-100
# --------------------------------------------------

def load_cifar100():
    """
    Load CIFAR-100 training and test datasets.

    The training dataset uses data augmentation.
    The test dataset uses only normalization.
    """

    train_transform, eval_transform = get_transforms(
        "CIFAR100"
    )

    train_dataset = datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR100(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=eval_transform,
    )

    return train_dataset, test_dataset


def load_cifar100_calibration():
    """
    Load the CIFAR-100 training images without augmentation.

    This dataset contains the same underlying training images
    in the same order as the dataset returned by load_cifar100().
    It allows prepare_data.py to select calibration examples
    without applying random training augmentations.

    This is not the official CIFAR-100 test dataset.
    """

    _, eval_transform = get_transforms("CIFAR100")

    calibration_dataset = datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=eval_transform,
    )

    return calibration_dataset


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # CIFAR-10
    cifar10_train, cifar10_test = load_cifar10()
    cifar10_calibration = load_cifar10_calibration()

    print("=== CIFAR-10 ===")
    print(f"Training images: {len(cifar10_train)}")
    print(f"Calibration dataset images: {len(cifar10_calibration)}")
    print(f"Test images: {len(cifar10_test)}")
    print(f"Image shape: {cifar10_train[0][0].shape}")

    print()

    # CIFAR-100
    cifar100_train, cifar100_test = load_cifar100()
    cifar100_calibration = load_cifar100_calibration()

    print("=== CIFAR-100 ===")
    print(f"Training images: {len(cifar100_train)}")
    print(f"Calibration dataset images: {len(cifar100_calibration)}")
    print(f"Test images: {len(cifar100_test)}")
    print(f"Image shape: {cifar100_train[0][0].shape}")
