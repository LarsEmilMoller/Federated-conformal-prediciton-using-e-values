#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:00:43 2026

@author: livboethrysoe
"""

"""
Empirical comparison of the algorithms FedCP²-QQ and FCeP-DP
on the Bike and Bio datasets.

Two comparisons are made:
    1. Empirical coverage as a function of the privacy parameter epsilon
       for a fixed number of clients K.
    2. Empirical coverage as a function of the number of clients K for
       a fixed total number of calibration points across all clients.

Both algorithms use the same calibration scores, client allocations,
test scores and training-based score bounds.
"""

import numpy as np
import matplotlib.pyplot as plt

from FedCP2_QQ_DP import select_gamma, fedcp2_qq
from FCeP_DP import fcep_dp_calibration, fcep_dp_e_values, acceptance
from experiment_FedCP2_QQ_DP import (
    load_scores,
    distribute_cores,
    load_coverage_matrix,
)


# -------------------------------------------------------------------------
# 1. Experimental settings
# -------------------------------------------------------------------------
DATASET = 'bike'            # 'bio', 'bike' or 'both'
ALPHA = 0.30
TOTAL_CALIBRATION = 1000    # Total calibration points across all clients
B = 100                    # Number of bins in FedCP²-QQ
SPLITS = 20
SEED = 42

# Privacy parameters.
EPSILONS = (5.0,)

# 30 logarithmically spaced values between 0.1 and 20, plus 1.0
# and the values in EPSILONS.
EPSILON_GRID = np.sort(np.r_[np.geomspace(0.1, 20, 30), 1.0, EPSILONS])
DELTA = 1e-12


# Federated settings for the two sweeps.
K_FIXED = 5

# Client counts must be divisors of TOTAL_CALIBRATION.
CLIENT_COUNTS = (2, 5, 10, 20, 25, 40, 50)

# Select which experiments to run (epsilon sweep or client sweep).
EXPERIMENTS = ('epsilon', 'clients')

METHODS = ('FedCP²-QQ', 'FCeP-DP')


# -------------------------------------------------------------------------
# 2. Prepare dataset splits
# -------------------------------------------------------------------------
def prepare_splits(data, K):
    """
    Allocate a fixed total number of calibration scores equally among K
    clients and retrieve the test scores and training-based score bound
    for each dataset split.

    Return a list of tuples (local_scores, test_scores, S_max),
    one for each dataset split.
    """
    if TOTAL_CALIBRATION % K != 0:
        raise ValueError('TOTAL_CALIBRATION must be divisible by K')

    n = TOTAL_CALIBRATION // K
    splits = []

    for split in range(SPLITS):
        # The seed and number of selected scores are independent of K.
        allocation_rng = np.random.default_rng(SEED + 100_000 + split)

        local_scores = distribute_scores(
            data['cal_scores'][split], K, n, allocation_rng
        )

        test_scores = data['test_scores'][split]
        S_max = float(data['S_max'][split])

        splits.append((local_scores, test_scores, S_max))

    return splits


# -------------------------------------------------------------------------
# 3. Compute empirical coverage
# -------------------------------------------------------------------------
def evaluate_both_algorithms(splits, eps, matrix):
    """
    Compute empirical coverage for both algorithms at a given epsilon.
    Return one coverage value per dataset split for each algorithm.
    Infeasible FedCP²-QQ configurations are represented by NaN.
    """
    try:
        qq_config = select_gamma(matrix, ALPHA, eps, B)
    except ValueError as exc:
        print(f'K={matrix.shape[0]}, epsilon={eps:g}: {exc}')
        qq_config = None

    coverage_results = {method: [] for method in METHODS}

    for split, (local_scores, test_scores, S_max) in enumerate(splits):

        # FedCP²-QQ: compute the private global quantile if feasible.
        if qq_config is not None:
            bin_edges = np.linspace(0, S_max, B + 1)
            qq_rng = np.random.default_rng(SEED + 300_000 + split)

            private_quantile = fedcp2_qq(
                local_scores, eps, bin_edges, qq_config, qq_rng
            )
            qq_coverage = np.mean(test_scores <= private_quantile)
        else:
            qq_coverage = np.nan

        coverage_results['FedCP²-QQ'].append(float(qq_coverage))

        # FCeP-DP: calibrate once and evaluate all test scores.
        fcep_rng = np.random.default_rng(SEED + 200_000 + split)

        fcep_calibration = fcep_dp_calibration(
            local_scores, S_max, eps, DELTA, fcep_rng
        )

        private_e_values = fcep_dp_e_values(
            test_scores, fcep_calibration
        )

        fcep_coverage = np.mean(acceptance(private_e_values, ALPHA))
        coverage_results['FCeP-DP'].append(float(fcep_coverage))

    return {
        method: np.asarray(coverages)
        for method, coverages in coverage_results.items()
    }


# -------------------------------------------------------------------------
# 4. Experiment 1: Epsilon sweep
# -------------------------------------------------------------------------
def epsilon_sweep(data):
    """
    Evaluate empirical coverage across epsilon with K and delta fixed.
    Return an array for each algorithm, with rows corresponding to
    epsilon values and columns corresponding to dataset splits.
    """
    splits = prepare_splits(data, K_FIXED)
    matrix = load_coverage_matrix(K_FIXED, TOTAL_CALIBRATION // K_FIXED)
    results = {method: [] for method in METHODS}

    for eps in EPSILON_GRID:
        coverage = evaluate_both_algorithms(splits, eps, matrix)

        for method in METHODS:
            results[method].append(coverage[method])

    results = {
        method: np.asarray(coverages)
        for method, coverages in results.items()
    }

    # Print feasibility summary for FedCP²-QQ.
    feasible = ~np.isnan(results['FedCP²-QQ']).all(axis=1)

    print(f'Feasible configurations: {np.sum(feasible)}/{len(EPSILON_GRID)}')
    print(f'Infeasible configurations: {np.sum(~feasible)}/{len(EPSILON_GRID)}')

    if np.any(feasible):
        print(f'Smallest feasible epsilon: {np.min(EPSILON_GRID[feasible]):.4f}')

    return results


# -------------------------------------------------------------------------
# 5. Experiment 2: Number of clients sweep
# -------------------------------------------------------------------------
def client_sweep(data):
    """
    Evaluate empirical coverage across client counts for selected epsilons.
    Keep the total calibration size and delta fixed.
    Return results indexed by epsilon and algorithm, with rows corresponding
    to client counts and columns corresponding to dataset splits.
    """
    results = {
        eps: {method: [] for method in METHODS}
        for eps in EPSILONS
    }

    for K in CLIENT_COUNTS:
        splits = prepare_splits(data, K)
        n = TOTAL_CALIBRATION // K

        print(f'K={K}, calibration points per client={n}')
        matrix = load_coverage_matrix(K, n)

        for eps in EPSILONS:
            coverage = evaluate_both_algorithms(splits, eps, matrix)

            for method in METHODS:
                results[eps][method].append(coverage[method])

    return {
        eps: {
            method: np.asarray(coverages)
            for method, coverages in values.items()
        }
        for eps, values in results.items()
    }


# -------------------------------------------------------------------------
# 6. Plotting
# -------------------------------------------------------------------------
def plot_epsilon_sweep(results):
    fig, axes = plt.subplots(
        1, len(results), figsize=(7.5 * len(results), 4.5)
    )

    fig.suptitle(
        'FedCP²-QQ vs FCeP-DP: Empirical coverage as a function of ε',
        fontweight='bold'
    )

    axes = np.atleast_1d(axes)

    for ax, (dataset, values) in zip(axes, results.items()):

        for method, coverages in values.items():
            ax.plot(
                EPSILON_GRID,
                np.mean(coverages, axis=1),
                marker='o',
                label=method
            )

        ax.axhline(
            1 - ALPHA, linestyle='--', color='black', label='Nominal'
        )

        ax.set_xscale('log')
        ax.set_xlabel('ε')
        ax.set_ylabel('Mean empirical coverage')
        ax.set_title(
            f'{dataset.capitalize()} (α={ALPHA:g}, K={K_FIXED}, δ={DELTA:g})',
            fontsize=12
        )

        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_client_sweep(results):
    # Produce one figure per epsilon, with a subplot for each dataset.
    for eps in EPSILONS:
        fig, axes = plt.subplots(
            1, len(results), figsize=(7.5 * len(results), 4.5)
        )

        fig.suptitle(
            'FedCP²-QQ vs FCeP-DP: Empirical coverage as a function of K',
            fontweight='bold'
        )

        axes = np.atleast_1d(axes)

        for ax, (dataset, values) in zip(axes, results.items()):

            for method, coverages in values[eps].items():
                ax.plot(
                    CLIENT_COUNTS,
                    np.mean(coverages, axis=1),
                    marker='o',
                    label=method
                )

            ax.axhline(
                1 - ALPHA, linestyle='--', color='black', label='Nominal'
            )

            ax.set_xticks(CLIENT_COUNTS)
            ax.set_xlabel('Number of clients K')
            ax.set_ylabel('Mean empirical coverage')
            ax.set_title(
                f'{dataset.capitalize()} (α={ALPHA:g}, ε={eps:g}, '
                f'δ={DELTA:g})',
                fontsize=12
            )

            ax.grid(alpha=0.25)
            ax.legend()

        fig.tight_layout()
        plt.show()


# -------------------------------------------------------------------------
# 7. Run the experiments
# -------------------------------------------------------------------------
def main():
    datasets = ('bio', 'bike') if DATASET == 'both' else (DATASET,)
    data = {dataset: load_scores(dataset) for dataset in datasets}
    results = {}

    if 'epsilon' in EXPERIMENTS:
        print('EXPERIMENT 1: Epsilon sweep')
        results['epsilon'] = {}

        for dataset in datasets:
            print(dataset)
            results['epsilon'][dataset] = epsilon_sweep(data[dataset])

        plot_epsilon_sweep(results['epsilon'])

    if 'clients' in EXPERIMENTS:
        print('\n\nEXPERIMENT 2: Number of clients sweep')
        results['clients'] = {}

        for dataset in datasets:
            print(dataset)
            results['clients'][dataset] = client_sweep(data[dataset])

        plot_client_sweep(results['clients'])

    return results


if __name__ == '__main__':
    results = main()

