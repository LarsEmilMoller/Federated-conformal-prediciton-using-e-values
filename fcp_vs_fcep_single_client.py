#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:44:32 2026

@author: livboethrysoe
"""


from pathlib import Path
import csv
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from load_data_v2 import load_cifar10, load_cifar10_calibration
from model_v2 import create_model


"""
Single-client alpha sweep: FCP versus FCeP.

Uses the existing single-client CNN checkpoint.
No model training or federated aggregation is performed.

Nonconformity score:
    S(x, y) = 1 - p_y(x)

FCP:
    C_alpha(x) = {y : S(x, y) <= q_alpha}

FCeP:
    E_y = (N + 1) S(x, y) / (S_G + S(x, y))
    C_alpha(x) = {y : E_y < 1 / alpha}
"""

# =========================================================
# Configuration
# =========================================================

SEED = 42
ALPHA_VALUES = [
    0.95, 0.90, 0.80, 0.70, 0.60, 0.50,
    0.40, 0.30, 0.20, 0.10, 0.05
]

NUM_CLASSES = 10
BATCH_SIZE = 128

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_PATH = RESULTS_DIR / "cnn_single_client_v3.pt"
RESULTS_PATH = RESULTS_DIR / "single_client_fcp_fcep_alpha_sweep.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# =========================================================
# Load the existing model
# =========================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Your create_model() does not accept num_classes.
    model = create_model("CIFAR10")

    # The single-client checkpoint stores a dictionary.
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Expected 'model_state_dict' in the checkpoint. "
            "Check the format of cnn_single_client_v3.pt."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Device: {DEVICE}")
    print(f"Model checkpoint: {MODEL_PATH}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if "validation_accuracy" in checkpoint:
        print(
            "Best validation accuracy: "
            f"{100 * checkpoint['validation_accuracy']:.2f}%"
        )

    return model


# =========================================================
# Load the single-client calibration and test datasets
# =========================================================

def load_datasets():
    """
    Reuse the existing data-loading functions.

    Assumes:
        load_cifar10() returns (training_dataset, test_dataset)
        load_cifar10_calibration() returns the deterministic
        version of the same 50,000 CIFAR-10 training examples.

    The calibration indices must match the original
    seed-42 split in prepare_data_single_client.py.
    """

    train_dataset, test_dataset = load_cifar10()
    calibration_base = load_cifar10_calibration()

    if len(train_dataset) != 50000:
        raise ValueError(
            f"Expected 50,000 training examples, "
            f"got {len(train_dataset)}."
        )

    if len(calibration_base) != len(train_dataset):
        raise ValueError(
            "Training and calibration datasets have different lengths."
        )

    # Check that both datasets use the same underlying label order.
    train_targets = getattr(train_dataset, "targets", None)
    calibration_targets = getattr(calibration_base, "targets", None)

    if train_targets is not None and calibration_targets is not None:
        if list(train_targets) != list(calibration_targets):
            raise ValueError(
                "Training and calibration label order differs. "
                "Cannot safely apply the original split indices."
            )

    # Reproduce the original seed-42 partition.
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(train_dataset),
        generator=generator
    ).tolist()

    # 40,000 training, 5,000 validation, 5,000 calibration.
    calibration_indices = indices[45000:50000]

    calibration_dataset = Subset(
        calibration_base,
        calibration_indices
    )

    if len(calibration_dataset) != 5000:
        raise ValueError(
            f"Expected 5,000 calibration examples, "
            f"got {len(calibration_dataset)}."
        )

    if len(test_dataset) != 10000:
        raise ValueError(
            f"Expected 10,000 test examples, "
            f"got {len(test_dataset)}."
        )

    return calibration_dataset, test_dataset


# =========================================================
# Calculate LAC scores and predicted class probabilities
# =========================================================

def compute_scores(model, dataset):
    """
    For each example, calculate its true-label score:
        S(x, y) = 1 - p_y(x)

    Also return the full class probability vector.
    """

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    true_label_scores = []
    labels_all = []
    probabilities_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels_device = labels.to(DEVICE)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            if probabilities.shape[1] != NUM_CLASSES:
                raise ValueError(
                    f"Expected {NUM_CLASSES} output classes, "
                    f"got {probabilities.shape[1]}."
                )

            true_class_probabilities = probabilities.gather(
                dim=1,
                index=labels_device.unsqueeze(1)
            ).squeeze(1)

            scores = 1.0 - true_class_probabilities

            true_label_scores.extend(scores.cpu().numpy())
            labels_all.extend(labels.numpy())
            probabilities_all.extend(probabilities.cpu().numpy())

    return (
        np.asarray(true_label_scores, dtype=np.float64),
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(probabilities_all, dtype=np.float64)
    )


# =========================================================
# FCP: single-client conformal quantile
# =========================================================

def fcp_quantile(calibration_scores, alpha):
    """
    Exact split-conformal order-statistic threshold.

    Rank = ceil((N + 1) * (1 - alpha)).

    If rank > N, return infinity.
    """

    n = len(calibration_scores)
    rank = math.ceil((n + 1) * (1.0 - alpha))

    if rank > n:
        return float("inf")

    sorted_scores = np.sort(calibration_scores)

    return float(sorted_scores[rank - 1])


# =========================================================
# FCeP: single-client e-values
# =========================================================

def fcep_prediction_sets(candidate_scores, calibration_score_sum,
                         n_calibration, alpha):
    """
    Apply the proposed FCeP formula:

        E_y = (N + 1) S(x, y) / (S_G + S(x, y))

    Include a class if E_y < 1 / alpha.
    """

    denominator = calibration_score_sum + candidate_scores

    if np.any(denominator == 0):
        raise ValueError(
            "FCeP encountered a zero denominator. "
            "The proposed e-value formula needs an explicit "
            "convention for this case."
        )

    e_values = (
        (n_calibration + 1) * candidate_scores / denominator
    )

    return e_values < (1.0 / alpha)


# =========================================================
# Evaluate both methods for one alpha
# =========================================================

def evaluate_alpha(
    test_probabilities,
    test_labels,
    calibration_scores,
    alpha
):
    n_calibration = len(calibration_scores)
    n_test = len(test_labels)

    calibration_score_sum = float(calibration_scores.sum())

    # FCP threshold for this alpha.
    q_alpha = fcp_quantile(calibration_scores, alpha)

    fcp_covered = 0
    fcep_covered = 0
    fcp_total_size = 0
    fcep_total_size = 0

    for start in range(0, n_test, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_test)

        probabilities = test_probabilities[start:end]
        labels = test_labels[start:end]

        # Scores for every possible class:
        # S(x, y) = 1 - p_y(x).
        candidate_scores = 1.0 - probabilities

        # FCP prediction sets.
        fcp_sets = candidate_scores <= q_alpha

        # FCeP prediction sets.
        fcep_sets = fcep_prediction_sets(
            candidate_scores,
            calibration_score_sum,
            n_calibration,
            alpha
        )

        row_indices = np.arange(len(labels))

        fcp_covered += int(
            fcp_sets[row_indices, labels].sum()
        )
        fcep_covered += int(
            fcep_sets[row_indices, labels].sum()
        )

        fcp_total_size += int(fcp_sets.sum())
        fcep_total_size += int(fcep_sets.sum())

    return {
        "alpha": alpha,
        "fcp_quantile": q_alpha,
        "fcp_coverage": fcp_covered / n_test,
        "fcp_avg_set_size": fcp_total_size / n_test,
        "fcep_coverage": fcep_covered / n_test,
        "fcep_avg_set_size": fcep_total_size / n_test,
        "calibration_size": n_calibration,
        "test_size": n_test
    }


# =========================================================
# Main experiment
# =========================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Single-client FCP versus FCeP alpha sweep")
    print("Number of clients: K = 1")
    print(f"Alpha values: {ALPHA_VALUES}")

    model = load_model()

    calibration_dataset, test_dataset = load_datasets()

    print(f"Calibration examples: {len(calibration_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # Calculate calibration scores once.
    calibration_scores, _, _ = compute_scores(
        model,
        calibration_dataset
    )

    # Calculate test probabilities once.
    _, test_labels, test_probabilities = compute_scores(
        model,
        test_dataset
    )

    results = []

    for alpha in ALPHA_VALUES:
        result = evaluate_alpha(
            test_probabilities,
            test_labels,
            calibration_scores,
            alpha
        )

        results.append(result)

        print(
            f"alpha={alpha:.2f} | "
            f"FCP coverage={result['fcp_coverage']:.4f}, "
            f"FCP size={result['fcp_avg_set_size']:.4f} | "
            f"FCeP coverage={result['fcep_coverage']:.4f}, "
            f"FCeP size={result['fcep_avg_set_size']:.4f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nAlpha sweep completed.")
    print(f"Results saved to:\n{RESULTS_PATH}")


if __name__ == "__main__":
    main()
