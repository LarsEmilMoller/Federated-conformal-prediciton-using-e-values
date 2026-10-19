#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:54:30 2026

@author: livboethrysoe
"""


import torch
import torch.nn as nn


"""
Convolutional neural network for CIFAR-10 and CIFAR-100.

Input:
    (batch_size, 3, 32, 32)

Output:
    (batch_size, num_classes) logits
"""


class CNN(nn.Module):
    """
    CNN for CIFAR-10 and CIFAR-100 classification.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 32 -> 16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 16 -> 8

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Return class logits for a batch of images.
        """

        x = self.features(x)
        logits = self.classifier(x)

        return logits


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
        A CNN with the corresponding number of output classes.
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


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    for dataset_name in ("CIFAR10", "CIFAR100"):

        model = create_model(dataset_name)

        images = torch.randn(4, 3, 32, 32)
        logits = model(images)

        print(f"\n=== {dataset_name} ===")
        print(f"Input shape: {images.shape}")
        print(f"Output shape: {logits.shape}")

        total_parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(f"Trainable parameters: {total_parameters:,}")
