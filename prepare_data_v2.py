#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:52:08 2026

@author: livboethrysoe
"""


import numpy as np
from torch.utils.data import Subset

from load_data_v2 import (
    load_cifar10,
    load_cifar10_calibration,
)


"""
Prepare CIFAR-10 for federated conformal prediction.

Paper-style non-IID partition:

Client 1: airplane, automobile
Client 2: bird, cat
Client 3: deer, dog
Client 4: frog, horse
Client 5: ship, truck

Each client receives:
    80% training data
    20% calibration data

Training data uses data augmentation.
Calibration data uses only tensor conversion and normalization.

The official CIFAR-10 test set remains untouched and is used
for final evaluation.
"""


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

CLIENT_CLASSES = {
    1: [0, 1],
    2: [2, 3],
    3: [4, 5],
    4: [6, 7],
    5: [8, 9],
}

TRAIN_RATIO = 0.8


# --------------------------------------------------
# Create client splits
# --------------------------------------------------

def create_client_splits(
    train_dataset,
    calibration_dataset,
    seed=SEED,
):
    """
    Create the five-client non-IID partition.

    Training and calibration datasets must contain the same
    underlying images in the same order, but may use different
    transformations.

    Parameters
    ----------
    train_dataset : torchvision dataset
        Training dataset with data augmentation.

    calibration_dataset : torchvision dataset
        Same training images, without random augmentation.

    seed : int
        Random seed used to shuffle and split the indices.

    Returns
    -------
    client_data : dict
        Each client has a training subset, a calibration subset,
        and a list of assigned class labels.
    """

    # Check that the datasets contain the same examples
    if len(train_dataset) != len(calibration_dataset):
        raise ValueError(
            "Training and calibration datasets must have "
            "the same length."
        )

    train_labels = np.asarray(train_dataset.targets)
    calibration_labels = np.asarray(calibration_dataset.targets)

    if not np.array_equal(train_labels, calibration_labels):
        raise ValueError(
            "Training and calibration datasets must have "
            "identical labels in the same order."
        )

    rng = np.random.default_rng(seed)

    labels = train_labels
    client_data = {}

    for client_id, classes in CLIENT_CLASSES.items():

        # Find all examples belonging to the client's classes
        indices = np.where(
            np.isin(labels, classes)
        )[0]

        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(TRAIN_RATIO * n_total)

        train_indices = indices[:n_train]
        calibration_indices = indices[n_train:]

        # The index sets are disjoint by construction
        client_data[client_id] = {
            "train": Subset(
                train_dataset,
                train_indices.tolist(),
            ),
            "calibration": Subset(
                calibration_dataset,
                calibration_indices.tolist(),
            ),
            "classes": classes.copy(),
        }

    return client_data


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    train_dataset, test_dataset = load_cifar10()
    calibration_dataset = load_cifar10_calibration()

    client_data = create_client_splits(
        train_dataset,
        calibration_dataset,
    )

    print("Federated CIFAR-10 partition\n")

    total_train = 0
    total_calibration = 0

    for client_id, data in client_data.items():

        print(f"Client {client_id}")
        print(f"Classes: {data['classes']}")
        print(f"Train dataset size: {len(data['train'])}")
        print(
            f"Calibration size: "
            f"{len(data['calibration'])}"
        )
        print()

        total_train += len(data["train"])
        total_calibration += len(data["calibration"])

    print(f"Total client training examples: {total_train}")
    print(f"Total client calibration examples: {total_calibration}")
    print(f"Official test set size: {len(test_dataset)}")
