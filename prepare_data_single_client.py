#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:57:45 2026

@author: livboethrysoe
"""


from torch.utils.data import Subset
import torch


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

TRAIN_SIZE = 40_000
VALIDATION_SIZE = 5_000
CALIBRATION_SIZE = 5_000


# --------------------------------------------------
# Single-client data preparation
# --------------------------------------------------

def prepare_data_single_client(
    train_dataset,
    calibration_dataset,
    seed=SEED,
):
    """
    Split the CIFAR-10 training data into disjoint training,
    validation, and calibration subsets.

    Parameters
    ----------
    train_dataset:
        Training dataset with training transformations.

    calibration_dataset:
        The same underlying examples in the same order,
        but with deterministic transformations.

    seed:
        Random seed used to create a reproducible split.

    Returns
    -------
    dict
        Dictionary containing 'train', 'validation',
        and 'calibration' subsets.
    """

    if len(train_dataset) != len(calibration_dataset):
        raise ValueError(
            "Training and calibration datasets must have "
            "the same number of examples."
        )

    if len(train_dataset) != (
        TRAIN_SIZE + VALIDATION_SIZE + CALIBRATION_SIZE
    ):
        raise ValueError(
            "Expected 50,000 CIFAR-10 training examples, "
            f"but received {len(train_dataset)}."
        )

    # Check that both dataset versions have the same labels
    # in the same order.
    train_targets = torch.as_tensor(train_dataset.targets)
    calibration_targets = torch.as_tensor(
        calibration_dataset.targets
    )

    if not torch.equal(train_targets, calibration_targets):
        raise ValueError(
            "Training and calibration datasets must contain "
            "the same labels in the same order."
        )

    # Generate a reproducible random permutation of indices.
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    ).tolist()

    train_indices = indices[:TRAIN_SIZE]

    validation_indices = indices[
        TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE
    ]

    calibration_indices = indices[
        TRAIN_SIZE + VALIDATION_SIZE:
    ]

    # Training uses the augmented dataset.
    train_subset = Subset(
        train_dataset,
        train_indices,
    )

    # Validation and calibration use deterministic transforms.
    validation_subset = Subset(
        calibration_dataset,
        validation_indices,
    )

    calibration_subset = Subset(
        calibration_dataset,
        calibration_indices,
    )

    # Verify that the subsets are disjoint.
    assert set(train_indices).isdisjoint(validation_indices)
    assert set(train_indices).isdisjoint(calibration_indices)
    assert set(validation_indices).isdisjoint(calibration_indices)

    print("Single-client data preparation complete.")
    print(f"Training examples: {len(train_subset)}")
    print(f"Validation examples: {len(validation_subset)}")
    print(f"Calibration examples: {len(calibration_subset)}")

    return {
        "train": train_subset,
        "validation": validation_subset,
        "calibration": calibration_subset,
    }


# --------------------------------------------------
# Main: sanity check
# --------------------------------------------------

if __name__ == "__main__":
    from load_data_v2 import (
        load_cifar10,
        load_cifar10_calibration,
    )

    train_dataset, test_dataset = load_cifar10()
    calibration_dataset = load_cifar10_calibration()

    print("Original dataset sizes:")
    print(f"Training dataset: {len(train_dataset)}")
    print(f"Calibration dataset: {len(calibration_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")

    data = prepare_data_single_client(
        train_dataset,
        calibration_dataset,
    )

    print("\nChecking subset overlap...")

    train_indices = set(data["train"].indices)
    validation_indices = set(data["validation"].indices)
    calibration_indices = set(data["calibration"].indices)

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(calibration_indices)
    assert validation_indices.isdisjoint(calibration_indices)

    assert len(
        train_indices | validation_indices | calibration_indices
    ) == len(train_dataset)

    print("PASS: Subsets are disjoint and cover all training examples.")

    print("\nChecking sample shapes...")

    for name, subset in data.items():
        image, label = subset[0]
        print(
            f"{name.capitalize()}: "
            f"image shape={tuple(image.shape)}, label={label}"
        )

        assert image.shape == (3, 32, 32)
        assert 0 <= int(label) < 10

    print("\nALL SANITY CHECKS PASSED")

