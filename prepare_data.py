#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:35:01 2026

@author: livboethrysoe
"""

import numpy as np
from torch.utils.data import Subset

from load_data import load_cifar10


"""
Prepare CIFAR-10 for federated conformal prediction.

Paper-style non-IID partition:

Client 0: airplane, automobile
Client 1: bird, cat
Client 2: deer, dog
Client 3: frog, horse
Client 4: ship, truck

Each client receives:
    80% training data
    20% calibration data

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

def create_client_splits(train_dataset, seed=SEED):
    """
    Create the paper's 5-client non-IID partition.

    Parameters
    ----------
    train_dataset : torchvision dataset

    Returns
    -------
    client_data : dict
    """

    rng = np.random.default_rng(seed)

    labels = np.asarray(train_dataset.targets)

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

        client_data[client_id] = {
            "train": Subset(train_dataset, train_indices),
            "calibration": Subset(
                train_dataset,
                calibration_indices
            ),
            "classes": classes,
        }

    return client_data


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    train_dataset, test_dataset = load_cifar10()

    client_data = create_client_splits(train_dataset)

    print("\nPaper-style CIFAR-10 partition\n")

    for client_id, data in client_data.items():

        print(f"Client {client_id}")
        print(f"Classes: {data['classes']}")
        print(f"Train size: {len(data['train'])}")
        print(
            f"Calibration size: "
            f"{len(data['calibration'])}"
        )
        print()

    print(
        f"Official test set size: "
        f"{len(test_dataset)}"
    )