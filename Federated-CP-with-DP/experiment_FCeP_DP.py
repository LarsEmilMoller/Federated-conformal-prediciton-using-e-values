#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:58:18 2026

@author: livboethrysoe
"""

"""
Empirical evaluation of the FCeP-DP algorithm on the Bike and Bio datasets.

Investigate sensitivity to delta, then visualize empirical
coverage across random splits using box plots at selected privacy levels.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from FCeP_DP import fcep_dp_calibration, fcep_dp_e_values, acceptance


# -------------------------------------------------------------------------
# 1. Experimental settings
# -------------------------------------------------------------------------
DATASET = 'bike'            # 'bio', 'bike' or 'both'
ALPHA = 0.10
K = 5
N_LOCAL = 200               # Calibration points per client
SPLITS = 20
SEED = 42

# Privacy parameters.
EPSILONS = (10.0, 5.0, 1.5)
DELTA_GRID = np.logspace(-1, -12, 12)  # 1e-1 to 1e-12
DELTA = 1e-12                           # Can be updated after the delta sweep

# Select which experiments to run (delta sweep or boxplots)
EXPERIMENTS = ('delta', 'boxplots')

CACHE_DIR = Path(__file__).resolve().parent / 'cached_data_DP'


# -------------------------------------------------------------------------
# 2. Load and distribute the calibration scores
# -------------------------------------------------------------------------
def load_scores(dataset):
    """
    Load the cached calibration and test scores and score bounds.
    """
    path = CACHE_DIR / f'{dataset}_scores.npz'

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def distribute_scores(cal_scores, K, n, rng):
    """
    Randomly select K*n calibration scores without replacement
    and allocate them to K clients with n scores per client.

    Return a (K, n) array, where each row contains the local
    calibration scores of one client.
    """
    scores = rng.choice(cal_scores, size=K * n, replace=False)

    return scores.reshape(K, n)


def prepare_splits(dataset):
    """
    Return a list of tuples (local_scores, test_scores, S_max), one for
    each dataset split, containing the local calibration scores of all K
    clients, all test scores and the training-based score bound S_max.

    The splits are independent of epsilon and delta and are therefore reused
    across all privacy parameter settings.
    """
    data = load_scores(dataset)
    splits = []

    for split in range(SPLITS):
        allocation_rng = np.random.default_rng(SEED + 100_000 + split)

        local_scores = distribute_scores(
            data['cal_scores'][split], K, N_LOCAL, allocation_rng
        )

        test_scores = data['test_scores'][split]

        # Retrieve the training-based score bound.
        S_max = float(data['S_max'][split])

        splits.append((local_scores, test_scores, S_max))

    return splits


# -------------------------------------------------------------------------
# 3. Compute empirical coverage
# -------------------------------------------------------------------------
def empirical_coverage(splits, eps, delta):
    """
    Compute empirical coverage of FCeP-DP for given epsilon and delta.

    For each dataset split, perform calibration once, compute the
    privatized federated e-values for all test scores and determine
    the proportion of test observations whose true labels are accepted.

    Return an array containing one empirical coverage value per split.
    """
    coverages = []

    for split, (local_scores, test_scores, S_max) in enumerate(splits):

        # Use the same split-specific seed across parameter settings to
        # reduce Monte Carlo variation when comparing privacy parameters.
        rng = np.random.default_rng(SEED + 200_000 + split)

        calibration = fcep_dp_calibration(
            local_scores, S_max, eps, delta, rng
        )

        e_values = fcep_dp_e_values(test_scores, calibration)

        coverage = np.mean(acceptance(e_values, ALPHA))
        coverages.append(float(coverage))

    return np.asarray(coverages)

# -------------------------------------------------------------------------
# 4. Experiment 1: Delta sweep
# -------------------------------------------------------------------------
def delta_sweep(splits):
    """
    Evaluate empirical coverage across delta for selected epsilons.
    """
    return {
        eps: np.array([
            empirical_coverage(splits, eps, delta) for delta in DELTA_GRID
        ])
        for eps in EPSILONS
    }


# -------------------------------------------------------------------------
# 5. Experiment 2: Box plot results
# -------------------------------------------------------------------------
def boxplot_results(splits):
    """
    Collect per-split coverage at the selected privacy levels.
    """
    results = {}

    for eps in EPSILONS:
        coverages = empirical_coverage(splits, eps, DELTA)

        results[f'ε = {eps:g}'] = coverages

    return results


# -------------------------------------------------------------------------
# 6. Plotting
# -------------------------------------------------------------------------
def plot_delta_sweep(results):
    fig, axes = plt.subplots(
        1, len(results), figsize=(7.5 * len(results), 4.5)
    )

    fig.suptitle(
        'FCeP-DP: Empirical coverage as a function of δ',
        fontweight='bold'
    )

    axes = np.atleast_1d(axes)

    for ax, (dataset, values) in zip(axes, results.items()):

        for eps, coverages in values.items():
            ax.plot(
                DELTA_GRID,
                np.mean(coverages, axis=1),
                marker='o',
                label=f'ε = {eps:g}'
            )

        ax.axhline(
            1 - ALPHA, linestyle='--', color='black', label='Nominal'
        )

        ax.set_xscale('log')
        ax.set_xlabel('δ')
        ax.set_ylabel('Mean empirical coverage')

        ax.set_title(
            f'{dataset.capitalize()} (α={ALPHA:g})',
            fontsize=12
        )

        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_boxplots(results):
    fig, axes = plt.subplots(
        1, len(results), figsize=(7.5 * len(results), 4.5)
    )

    fig.suptitle(
        'FCeP-DP: Empirical coverage across random splits',
        fontweight='bold'
    )

    axes = np.atleast_1d(axes)

    rng = np.random.default_rng(SEED + 9_999)
    colors = ['#b3cde3', '#decbe4', '#fed9a6']

    for ax, (dataset, values) in zip(axes, results.items()):

        labels = list(values)
        positions = np.arange(len(labels), 0, -1)

        boxes = ax.boxplot(
            list(values.values()),
            positions=positions,
            vert=False,
            patch_artist=True,
            widths=0.4,
            whis=(0, 100),
            showfliers=False
        )

        for box, color in zip(boxes['boxes'], colors):
            box.set_facecolor(color)

        for position, coverages in zip(positions, values.values()):

            ax.scatter(
                coverages,
                position + rng.normal(0, 0.04, len(coverages)),
                s=15,
                alpha=0.55,
                color='#555555'
            )

            ax.scatter(
                np.mean(coverages),
                position,
                s=58,
                facecolor='white',
                edgecolor='black',
                zorder=4
            )

        ax.axvline(1 - ALPHA, linestyle='--', color='black')

        ax.set_yticks(positions, labels)
        ax.set_xlabel('Coverage')

        min_coverage = min(min(v) for v in values.values())

        ax.set_xlim(
            max(0, min(1 - ALPHA - 0.06, min_coverage - 0.02)),
            1.005
        )

        ax.set_title(
            f'{dataset.capitalize()} (α={ALPHA:g}, δ={DELTA:g})',
            fontsize=12
        )

        ax.grid(axis='x', alpha=0.25)

    fig.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# 7. Run the experiments
# -------------------------------------------------------------------------
def main():
    datasets = ('bio', 'bike') if DATASET == 'both' else (DATASET,)

    prepared = {
        dataset: prepare_splits(dataset) for dataset in datasets
    }

    results = {}

    if 'delta' in EXPERIMENTS:
        print('Experiment 1: Delta sweep')

        results['delta'] = {
            dataset: delta_sweep(splits)
            for dataset, splits in prepared.items()
        }

        plot_delta_sweep(results['delta'])

    if 'boxplots' in EXPERIMENTS:
        print(f'Experiment 2: Box plots (delta={DELTA:g})')

        results['boxplots'] = {
            dataset: boxplot_results(splits)
            for dataset, splits in prepared.items()
        }

        plot_boxplots(results['boxplots'])

        for dataset, values in results['boxplots'].items():
            for method, coverages in values.items():
                print(
                    f'{dataset:<5} {method:<13}: '
                    f'{np.mean(coverages):.4f} ± {np.std(coverages):.4f}'
                )

    return results


if __name__ == '__main__':
    results = main()