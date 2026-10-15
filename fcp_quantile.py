#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:53:38 2026

@author: livboethrysoe
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from load_data import load_cifar10
from prepare_data import create_client_splits
from model import create_model


# ==================================================
# Configuration
# ==================================================

ALPHA = 0.10
BATCH_SIZE = 128
DATASET_NAME = "CIFAR10"

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    SCRIPT_DIR
    / "results"
    / "fedavg_cifar10.pt"
)


# ==================================================
# Device
# ==================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# ==================================================
# Load trained FedAvg model
# ==================================================

def load_global_model():

    print("\nLoading trained FedAvg model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model = create_model(DATASET_NAME)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ==================================================
# Compute LAC nonconformity scores
# ==================================================

def compute_lac_scores(
    model,
    calibration_dataset,
):
    """
    LAC score:

    score = 1 - p(true class)
    """

    loader = DataLoader(
        calibration_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    scores = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            true_probs = probabilities[
                torch.arange(
                    labels.size(0),
                    device=DEVICE,
                ),
                labels,
            ]

            batch_scores = (
                1.0 - true_probs
            )

            scores.extend(
                batch_scores.cpu().numpy()
            )

    return np.asarray(scores)


# ==================================================
# Quantile computation
# ==================================================

def compute_quantile(
    scores,
    alpha,
):
    """
    Standard split conformal quantile.
    """

    scores = np.sort(scores)

    n = len(scores)

    k = int(
        np.ceil(
            (n + 1) * (1 - alpha)
        )
    ) - 1

    k = min(k, n - 1)

    return scores[k]


# ==================================================
# Construct prediction set
# ==================================================

def construct_prediction_set(
    model,
    image,
    q_alpha,
):

    with torch.no_grad():

        image = image.unsqueeze(0).to(DEVICE)

        logits = model(image)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    pred_set = []

    for class_id in range(
        len(probabilities)
    ):

        score = (
            1.0
            - probabilities[class_id]
        )

        if score <= q_alpha:
            pred_set.append(class_id)

    return pred_set


# ==================================================
# Evaluate coverage and set size
# ==================================================

def evaluate_fcp(
    model,
    test_dataset,
    q_alpha,
):

    covered = 0
    total = 0

    prediction_set_sizes = []

    print(
        "\nEvaluating conformal predictor..."
    )

    for image, true_label in test_dataset:

        pred_set = construct_prediction_set(
            model,
            image,
            q_alpha,
        )

        prediction_set_sizes.append(
            len(pred_set)
        )

        if true_label in pred_set:
            covered += 1

        total += 1

    coverage = covered / total

    average_set_size = np.mean(
        prediction_set_sizes
    )

    return (
        coverage,
        average_set_size,
    )


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    print("\nSTARTING FCP-QUANTILE")

    model = load_global_model()

    print("\nLoading CIFAR-10...")

    train_dataset, test_dataset = (
        load_cifar10()
    )

    print("Creating client splits...")

    client_data = create_client_splits(
        train_dataset
    )

    print(
        "\nComputing calibration scores..."
    )

    all_scores = []

    for client_id, data in client_data.items():

        if "calibration" in data:
            calibration_dataset = data[
                "calibration"
            ]
        elif "cal" in data:
            calibration_dataset = data[
                "cal"
            ]
        else:
            raise KeyError(
                f"No calibration split found "
                f"for client {client_id}"
            )

        scores = compute_lac_scores(
            model,
            calibration_dataset,
        )

        print(
            f"Client {client_id}: "
            f"{len(scores)} scores"
        )

        all_scores.extend(scores)

    all_scores = np.asarray(
        all_scores
    )

    print(
        f"\nTotal calibration scores: "
        f"{len(all_scores)}"
    )

    q_alpha = compute_quantile(
        all_scores,
        ALPHA,
    )

    print(
        f"\nAlpha = {ALPHA}"
    )

    print(
        f"q_alpha = {q_alpha:.6f}"
    )

    coverage, avg_set_size = evaluate_fcp(
        model,
        test_dataset,
        q_alpha,
    )

    print("\nResults")
    print("-" * 40)

    print(
        f"Coverage: "
        f"{100 * coverage:.2f}%"
    )

    print(
        f"Average prediction set size: "
        f"{avg_set_size:.2f}"
    )

    print("\nFinished.")