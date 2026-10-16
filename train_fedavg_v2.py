#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:01:03 2026

@author: livboethrysoe
"""

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data_v2 import (
    load_cifar10,
    load_cifar10_calibration,
)
from prepare_data_v2 import create_client_splits
from model_v2 import create_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_NAME = "CIFAR10"

NUM_ROUNDS = 50
LOCAL_EPOCHS = 2

BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = RESULTS_DIR / "fedavg_cifar10.pt"


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# --------------------------------------------------
# Local client training
# --------------------------------------------------

def train_client(model, dataset):
    """
    Train a local client model for the specified number
    of epochs and return its weights and average batch loss.
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()

    total_loss = 0.0
    num_batches = 0

    for epoch in range(LOCAL_EPOCHS):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches

    # Copy weights to CPU before returning them.
    client_weights = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    return client_weights, average_loss


# --------------------------------------------------
# FedAvg aggregation
# --------------------------------------------------

def average_weights(client_weight_dicts, client_sizes):
    """
    Aggregate client model weights using sample-size-weighted
    FedAvg. Non-floating-point buffers are copied from the
    first client rather than averaged.
    """

    if not client_weight_dicts:
        raise ValueError("No client weights were provided.")

    if len(client_weight_dicts) != len(client_sizes):
        raise ValueError(
            "The number of client models and client sizes must match."
        )

    if any(size <= 0 for size in client_sizes):
        raise ValueError("All client training sizes must be positive.")

    total_size = sum(client_sizes)

    aggregated_weights = deepcopy(client_weight_dicts[0])

    for parameter_name in aggregated_weights:
        first_tensor = client_weight_dicts[0][parameter_name]

        if torch.is_floating_point(first_tensor):
            weighted_sum = torch.zeros_like(first_tensor)

            for client_weights, client_size in zip(
                client_weight_dicts,
                client_sizes,
            ):
                weighted_sum += (
                    client_weights[parameter_name]
                    * (client_size / total_size)
                )

            aggregated_weights[parameter_name] = weighted_sum

        else:
            # For example, BatchNorm's num_batches_tracked.
            aggregated_weights[parameter_name] = first_tensor.clone()

    return aggregated_weights


# --------------------------------------------------
# Global model evaluation
# --------------------------------------------------

def evaluate(model, dataset):
    """
    Evaluate a model and return its accuracy as a proportion
    between 0 and 1.
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate on an empty dataset.")

    return correct / total


# --------------------------------------------------
# FedAvg training
# --------------------------------------------------

def train_fedavg(client_data, test_dataset):
    """
    Train a global model using FedAvg.

    Print client training losses and global accuracy
    in round 1 and every 10 rounds thereafter.
    """

    global_model = create_model(DATASET_NAME).to(DEVICE)

    for round_num in range(1, NUM_ROUNDS + 1):

        print_round = (
            round_num == 1 or round_num % 10 == 0
        )

        if print_round:
            print(f"\nCommunication Round {round_num}/{NUM_ROUNDS}")

        client_weights = []
        client_sizes = []

        # Train each client locally.
        for client_id, data in client_data.items():

            if print_round:
                print(f"Training Client {client_id}")

            client_model = create_model(DATASET_NAME).to(DEVICE)
            client_model.load_state_dict(global_model.state_dict())

            weights, loss = train_client(
                client_model,
                data["train"],
            )

            client_weights.append(weights)
            client_sizes.append(len(data["train"]))

            if print_round:
                print(f"Average training loss: {loss:.4f}")

        # Aggregate the client models.
        global_weights = average_weights(
            client_weights,
            client_sizes,
        )

        global_model.load_state_dict(global_weights)

        # Evaluate the global model after aggregation.
        global_accuracy = evaluate(
            global_model,
            test_dataset,
        )

        if print_round:
            print(f"Global accuracy: {100 * global_accuracy:.2f}%")

    return global_model



# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("\nLoading CIFAR-10...")

    train_dataset, test_dataset = load_cifar10()
    calibration_dataset = load_cifar10_calibration()

    print("Creating client splits...")

    client_data = create_client_splits(
        train_dataset,
        calibration_dataset,
    )

    print("Starting FedAvg training...")

    global_model = train_fedavg(
        client_data,
        test_dataset,
    )

    # Save the final global model's state dictionary.
    torch.save(
        global_model.state_dict(),
        MODEL_PATH,
    )

    print(f"\nModel saved to:\n{MODEL_PATH}")

    # Report final test accuracy.
    final_accuracy = evaluate(
        global_model,
        test_dataset,
    )

    print(
        f"\nFinal global test accuracy: "
        f"{100 * final_accuracy:.2f}%"
    )

