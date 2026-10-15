#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:42:35 2026

@author: livboethrysoe
"""


from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data import load_cifar10
from prepare_data import create_client_splits
from model import create_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_NAME = "CIFAR10"

NUM_ROUNDS = 50
LOCAL_EPOCHS = 2

BATCH_SIZE = 128
LEARNING_RATE = 0.001

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
    Train a local client model.
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for epoch in range(LOCAL_EPOCHS):

        running_loss = 0.0

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return model.state_dict()


# --------------------------------------------------
# FedAvg aggregation
# --------------------------------------------------

def average_weights(client_weights):
    """
    Standard FedAvg parameter averaging.
    """

    averaged_weights = deepcopy(
        client_weights[0]
    )

    for key in averaged_weights.keys():

        for i in range(1, len(client_weights)):

            averaged_weights[key] += (
                client_weights[i][key]
            )

        averaged_weights[key] /= len(
            client_weights
        )

    return averaged_weights


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, dataset):

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

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


# --------------------------------------------------
# FedAvg training
# --------------------------------------------------

def train_fedavg(client_data, test_dataset):

    global_model = create_model(
        DATASET_NAME
    ).to(DEVICE)

    for round_number in range(NUM_ROUNDS):

        print(
            f"\nCommunication Round "
            f"{round_number + 1}/{NUM_ROUNDS}"
        )

        client_weights = []

        for client_id in client_data:

            print(
                f"Training Client {client_id}"
            )

            local_model = deepcopy(
                global_model
            )

            local_state_dict = train_client(
                local_model,
                client_data[client_id]["train"],
            )

            client_weights.append(
                local_state_dict
            )

        averaged_weights = average_weights(
            client_weights
        )

        global_model.load_state_dict(
            averaged_weights
        )

        accuracy = evaluate(
            global_model,
            test_dataset,
        )

        print(
            f"Global accuracy: "
            f"{100*accuracy:.2f}%"
        )

    return global_model


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("\nLoading CIFAR-10...")

    train_dataset, test_dataset = (
        load_cifar10()
    )

    print("Creating client splits...")

    client_data = create_client_splits(
        train_dataset
    )

    print("Starting FedAvg training...")

    global_model = train_fedavg(
        client_data,
        test_dataset,
    )

    torch.save(
        global_model.state_dict(),
        MODEL_PATH,
    )

    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )

    final_accuracy = evaluate(
        global_model,
        test_dataset,
    )

    print(
        f"\nFinal test accuracy: "
        f"{100*final_accuracy:.2f}%"
    )