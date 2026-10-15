#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:38:55 2026

@author: livboethrysoe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Convolutional neural network for CIFAR-10 and CIFAR-100.

Input:
    (batch_size, 3, 32, 32)

Output:
    (batch_size, num_classes) logits
"""


class CNN(nn.Module):
    """
    Simple CNN for image classification.
    """

    def __init__(self, num_classes):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            128 * 8 * 8,
            256,
        )

        self.dropout = nn.Dropout(
            p=0.5,
        )

        self.fc2 = nn.Linear(
            256,
            num_classes,
        )

    def forward(self, x):

        # (3, 32, 32) -> (32, 32, 32)
        x = F.relu(self.conv1(x))

        # (32, 32, 32) -> (64, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))

        # (64, 16, 16) -> (128, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Return logits
        return x


def create_model(dataset_name):
    """
    Create a CNN with the correct number of output classes.

    Parameters
    ----------
    dataset_name : str
        "CIFAR10" or "CIFAR100"

    Returns
    -------
    CNN
    """

    if dataset_name == "CIFAR10":
        num_classes = 10

    elif dataset_name == "CIFAR100":
        num_classes = 100

    else:
        raise ValueError(
            "dataset_name must be 'CIFAR10' or 'CIFAR100'."
        )

    return CNN(num_classes=num_classes)


if __name__ == "__main__":

    dataset_name = "CIFAR10"

    model = create_model(dataset_name)

    images = torch.randn(
        4,
        3,
        32,
        32,
    )

    logits = model(images)

    print(f"Dataset: {dataset_name}")
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {logits.shape}")

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTrainable parameters: {total_parameters:,}")

    print("\nParameters per layer:")

    for name, parameter in model.named_parameters():
        print(f"{name}: {parameter.numel():,}")