#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:54:16 2026

@author: livboethrysoe
"""

"""
Empirical evaluation of FedCP-QQ and FedCP²-QQ from
"One-Shot Federated Conformal Prediction" by Humbert et al. (2023).

Empirical coverage on the Bike and Bio datasets is visualized using
box plots. The experimental settings are adapted from Appendix B.2.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from FedCP2_QQ_DP import (
    calc_matrix_M,
    find_l_star_k_star,
    select_gamma,
    fedcp_qq,
    fedcp2_qq,
)

# -------------------------------------------------------------------------
# 1. Experimental parameters
# -------------------------------------------------------------------------
DATASET = 'bike'            # 'bio', 'bike' or 'both'
ALPHA = 0.10
EPSILONS = (10.0, 5.0, 1.5)
M = 5
N = 200
B = 100
SPLITS = 20
SEED = 42

CACHE_DIR = Path(__file__).resolve().parent / 'cached_data_DP'


# -------------------------------------------------------------------------
# 2. Loading and allocation of scores
# -------------------------------------------------------------------------
def load_scores(dataset):
    """
    Load the cached calibration and test scores and score bounds.
    """
    path = CACHE_DIR / f'{dataset}_scores.npz'

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def distribute_scores(cal_scores, m, n, rng):
    """
    Randomly select m*n calibration scores without replacement
    and allocate them to m clients with n scores per client.

    Return an (m, n) array, where each row contains the local
    calibration scores of one client.
    """
    scores = rng.choice(cal_scores, size=m*n, replace=False)

    return scores.reshape(m, n)


# -------------------------------------------------------------------------
# 3. Computation and caching of the coverage matrix
# -------------------------------------------------------------------------
def load_coverage_matrix(m, n):
    """
    Compute or load the coverage matrix M.

    Following the authors' implementation, M has dimensions (m, n),
    with rows indexed by k and columns indexed by ell.
    """
    folder = CACHE_DIR / 'coverage_matrices'
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f'M_hypergeo_v1_m{m}_n{n}.npy'

    # Load the cached matrix if it exists; otherwise compute it.
    if path.exists():
        matrix = np.load(path, allow_pickle=False)
    else:
        matrix = calc_matrix_M(m, n, 0.5)

    # Check for numerical errors in the coverage matrix.
    if not np.isfinite(matrix).all():
        raise ValueError('Coverage matrix contains non-finite values')

    # Cache the matrix if newly computed.
    if not path.exists():
        np.save(path, matrix)

    return matrix


# -------------------------------------------------------------------------
# 4. Empirical coverage
# -------------------------------------------------------------------------
def empirical_coverage(test_scores, threshold):
    """
    Compute the proportion of test scores not exceeding the threshold.
    """
    return float(np.mean(test_scores <= threshold))


# -------------------------------------------------------------------------
# 5. Execution of the experiment
# -------------------------------------------------------------------------
def run_experiment(dataset_name, n_splits, seed, *, m, n, B, alpha, epsilons):
    """
    Evaluate the empirical coverage of FedCP-QQ and FedCP²-QQ
    across multiple dataset splits, using different values of epsilon
    for FedCP²-QQ.
    """

    # Load the precomputed scores and coverage matrix.
    data = load_scores(dataset_name)
    matrix = load_coverage_matrix(m, n)

    # Determine the quantile orders for the non-private FedCP-QQ.
    ell, k = find_l_star_k_star(matrix, 1 - alpha)

    print(
        f'Non-private: ell={ell}, k={k}, '
        f'coverage={matrix[k-1, ell-1]:.6f}'
    )

    # Select the optimal gamma and corresponding configuration for each
    # epsilon. Skip epsilon values for which no feasible gamma is found.
    configs = {}

    for eps in epsilons:
        try:
            configs[eps] = select_gamma(matrix, alpha, eps, B)
            print(f'epsilon={eps:g}: {configs[eps]}')
        except ValueError as exc:
            print(f'Skipping epsilon={eps:g}: {exc}')

    # Initialize a dictionary to store one empirical coverage value
    # per dataset split for each algorithm configuration.
    results = {
        'FedCP-QQ': [],
        **{f'ε = {eps:g}': [] for eps in configs}
    }

    # Initialize the random number generator for the exponential mechanism.
    dp_rng = np.random.default_rng(seed)

    # Evaluate all algorithm configurations on each dataset split.
    for split in range(n_splits):

        # Randomly allocate n calibration scores to each of the m clients
        # and extract the corresponding test scores.
        # All algorithm configurations use the same scores within each split.
        allocation_rng = np.random.default_rng(seed + 100_000 + split)

        clients = distribute_scores(
            data['cal_scores'][split], m, n, allocation_rng
        )

        test = data['test_scores'][split]

        # Compute the global quantile using FedCP-QQ
        # and store the resulting empirical coverage.
        threshold = fedcp_qq(clients, ell, k)

        results['FedCP-QQ'].append(
            empirical_coverage(test, threshold)
        )

        # Retrieve the training-based score bound and construct
        # B equally spaced bins.
        S_max = float(data['S_max'][split])
        edges = np.linspace(0, S_max, B + 1)

        # For each feasible epsilon, compute the private global quantile
        # using FedCP²-QQ and store the resulting empirical coverage.
        for eps, config in configs.items():
            threshold = fedcp2_qq(
                clients, eps, edges, config, dp_rng
            )

            results[f'ε = {eps:g}'].append(
                empirical_coverage(test, threshold)
            )

        print(f'{dataset_name}: split {split+1}/{n_splits}')

    # Return the empirical coverage values across all dataset splits.
    return results


# -------------------------------------------------------------------------
# 6. Box plots of empirical coverage
# -------------------------------------------------------------------------
def plot_results(results, alpha, seed):
    """
    Produce box plots of empirical coverage across dataset splits.
    """
    fig, axes = plt.subplots(
        1, len(results), figsize=(7.5 * len(results), 4.5)
    )

    axes = np.atleast_1d(axes)

    rng = np.random.default_rng(seed + 9_999)
    colors = ['#8da0cb', '#b3cde3', '#decbe4', '#fed9a6']

    for ax, (dataset, values) in zip(axes, results.items()):

        labels = list(values)
        positions = np.arange(len(labels), 0, -1)

        # Create horizontal box plots.
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

        # Display individual splits and mean empirical coverage.
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

        # Indicate the nominal coverage level.
        ax.axvline(1 - alpha, linestyle='--', color='black')

        ax.set_yticks(positions, labels)
        ax.set_xlabel('Coverage')

        min_coverage = min(min(v) for v in values.values())
        ax.set_xlim(
            max(0, min(1 - alpha - 0.06, min_coverage - 0.02)),
            1.005
        )

        # Subtitles for each dataset.
        ax.set_title(
            f'{dataset.capitalize()} (α={alpha:g})'
        )

        ax.grid(axis='x', alpha=0.25)

    # Overall title.
    fig.suptitle(
        'FedCP-QQ and FedCP²-QQ: Empirical coverage across random splits',
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# 7. Main
# -------------------------------------------------------------------------
def main():
    """
    Run the experiments and display the empirical coverage results.
    """
    names = ('bio', 'bike') if DATASET == 'both' else (DATASET,)

    all_results = {}

    for name in names:
        all_results[name] = run_experiment(
            name,
            SPLITS,
            SEED,
            m=M,
            n=N,
            B=B,
            alpha=ALPHA,
            epsilons=EPSILONS
        )

    plot_results(all_results, ALPHA, SEED)

    # Print the mean and standard deviation of empirical coverage.
    for name, values in all_results.items():
        for method, coverages in values.items():
            print(
                f'{name:<5} {method:<13}: '
                f'{np.mean(coverages):.4f} ± {np.std(coverages):.4f}'
            )

    return all_results


if __name__ == '__main__':
    results = main()