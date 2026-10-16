#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:00:46 2026

@author: livboethrysoe
"""


from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data_v2 import load_cifar10, load_cifar10_calibration
from prepare_data_single_client import prepare_data_single_client
from model_v2 import create_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_NAME = "CIFAR10"

NUM_EPOCHS = 100

BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "cnn_single_client_v3.pt"


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
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += batch_size

    accuracy = correct / total
    average_loss = total_loss / total

    return accuracy, average_loss


# --------------------------------------------------
# Training
# --------------------------------------------------

def train():
    set_seed(SEED)

    # Load CIFAR-10 datasets.
    train_dataset, test_dataset = load_cifar10()
    calibration_dataset = load_cifar10_calibration()

    # Create disjoint training, validation, and calibration subsets.
    data = prepare_data_single_client(
        train_dataset,
        calibration_dataset,
        seed=SEED,
    )

    train_subset = data["train"]
    validation_subset = data["validation"]
    calibration_subset = data["calibration"]

    print("\nDataset sizes:")
    print(f"Training: {len(train_subset)}")
    print(f"Validation: {len(validation_subset)}")
    print(f"Calibration: {len(calibration_subset)}")
    print(f"Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = create_model(DATASET_NAME).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.1,
    )

    best_validation_accuracy = float("-inf")
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_loss = 0.0
        total_examples = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        average_training_loss = total_loss / total_examples

        # Select the checkpoint using validation data only.
        validation_accuracy, validation_loss = evaluate(
            model,
            validation_subset,
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset": DATASET_NAME,
                    "epoch": best_epoch,
                    "validation_accuracy": best_validation_accuracy,
                    "seed": SEED,
                },
                MODEL_PATH,
            )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Training Loss: {average_training_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {100 * validation_accuracy:.2f}% | "
            f"Best: {100 * best_validation_accuracy:.2f}%"
        )

        scheduler.step()

    print(f"\nBest validation accuracy: {100 * best_validation_accuracy:.2f}%")
    print(f"Best epoch: {best_epoch}")
    print(f"Best model saved to:\n{MODEL_PATH}")

    return MODEL_PATH


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    train()
