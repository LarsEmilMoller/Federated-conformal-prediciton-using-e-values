#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:33:00 2026

@author: livboethrysoe
"""


"""
Reproduces the empirical results for the FedCP²-QQ algorithm from
"One-Shot Federated Conformal Prediction" (Humbert et al., ICML 2023).

Specifically, the script reproduces the empirical coverage box plots
in Figure 18 using the Bike and Bio datasets.

The implementation follows the paper, with minor adjustments where
implementation details are not specified. These include handling
negative CQR scores and quantile levels q >= 1. Additionally, the
authors' coverage-matrix code is adjusted for numerical stability.
These adjustments are not covered by the paper's stated guarantees.

Configure the experiment settings below and run the script.

Reference: https://github.com/pierreHmbt/FedCP-QQ
"""


from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom
import scipy.special as sc
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from quantile_forest import RandomForestQuantileRegressor
from sklearn.preprocessing import StandardScaler


# ========================================================================
# EXPERIMENT SETTINGS — change these values, then rerun the script
# ========================================================================
DATASET = 'bike'            # 'bike', 'bio', or 'both'

ALPHA = 0.10                # Target miscoverage; coverage is 1 - ALPHA
EPSILONS = (10.0, 5.0, 1.0) # Privacy levels to compare
M = 5                       # Number of clients
N = 200                     # Calibration samples per client
B = 100                     # Number of bins

SPLITS = 20                 # Number of random train/cal/test splits
TREES = 1000                # Trees in the quantile random forest
SEED = 42

# ========================================================================


# -------------------------------------------------------------------------
# 1. Algorithm 1: FedCP-QQ (Non-private Quantile-of-Quantiles)
# -------------------------------------------------------------------------
"""
The functions sum_hypergeo and calc_matrix_M are based on the authors'
implementation in class_CP_QQ.py:
https://github.com/pierreHmbt/FedCP-QQ/blob/main/class_CP_QQ.py

One minor modification was made: two CDF subtractions were replaced with
mathematically equivalent PMF sums to avoid catastrophic cancellation
when cumulative probabilities are close to 1.
"""
def sum_hypergeo(aa, bb, n, t, epsi=0):
    m = len(aa)
    # ==========================
    rv1 = binom(n, t)
    prob = []
    for i in range(m):
        prob.append(np.sum(rv1.pmf(np.arange(aa[i], bb[i] + 1))))

    r1 = 1.
    for i in range(m):
        r1 *= prob[i]
    # ==========================

    # ==========================
    rv2 = binom(m*n, t)
    # ==========================

    # ==========================
    gen = []
    for i in range(m):
        # tt = 1 - ( rv1.cdf(aa[i]-1) + (1 - rv1.cdf(bb[i])) )
        tt = np.sum(rv1.pmf(np.arange(aa[i], bb[i] + 1)))

        pp = []
        for j in np.arange(n, -1, -1):
            if j<=bb[i] and j>=aa[i]:
                pp.append(rv1.pmf(j)/tt)
            else:
                pp.append(0)
        gen.append(np.poly1d(pp))

    poly = gen[0]
    for i in range(1, len(gen)):
        poly = np.polymul(poly, gen[i])
    # ==========================

    s = 0
    for l in range(np.sum(aa), np.sum(bb) + 1):
        r2 = rv2.pmf(l)
        if r2>epsi:
            r3 = poly[l]
            s += r3*r1/r2
    return s


def calc_matrix_M(m, n, t, epsi=.0, mid=False):
    A = np.zeros((m, n))
    for aa in np.arange((m//2)*mid, m):
        for bb in np.arange((n//2)*mid, n):
            r = aa+1
            k = bb+1
            s = 0
            for j in range(r, m+1):
                a_range = np.ones(m, dtype=int)*k
                a_range[j:] = 0
                b_range = np.ones(m, dtype=int)*n
                b_range[j:] = k-1
                s += sum_hypergeo(a_range, b_range, n, .5, epsi)*sc.binom(m, j)

            A[aa, bb] = 1-s/(m*n+1)
    return A


def find_l_star_k_star(M: np.ndarray, target_cov: float) -> tuple[int, int]:
    """
    Returns the quantile orders with the smallest M_{l,k} >= 1-alpha.
    """

    valid = np.argwhere(M >= target_cov - 1e-12)  # Small tolerance to avoid floating-point rounding errors

    if len(valid) == 0:
        raise ValueError(f'No quantile orders achieve coverage {target_cov:.6f}')

    ell, k = min(valid, key=lambda idx: M[tuple(idx)])

    return int(ell + 1), int(k + 1)


def fedcp_qq(local_scores: np.ndarray, ell: int, k: int) -> float:
    """
    Returns the global quantile-of-quantiles.
    """
    client_quantiles = np.partition(local_scores, ell - 1, axis=1)[:, ell - 1]
    server_quantile = np.partition(client_quantiles, k - 1)[k - 1]

    return float(server_quantile)

# -------------------------------------------------------------------------
# 2. Algorithm 2: Differentially private quantile (exponential mechanism)
# -------------------------------------------------------------------------
def dp_quantile(scores: np.ndarray, q: float, eps: float,
                bin_edges: np.ndarray, rng: np.random.Generator) -> float:
    """
    Returns a differentially private quantile using the exponential mechanism.

    To allow Algorithm 2 to be used within Algorithm 3 when q >= 1,
    we use the limit as q approaches 1 from below (q -> 1^-).

    This extension is not part of Algorithm 2 as defined in the paper.
    """
    scores = np.asarray(scores, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)

    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError('scores must be a nonempty finite 1D array')
    if len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('bin_edges must be strictly increasing, finite')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('epsilon must be positive')
    if np.any(scores < edges[0]) or np.any(scores > edges[-1]):
        raise ValueError('scores outside the chosen bin range')

    upper_edges = edges[1:]

    # Score at the left endpoint belongs to the first bin.
    discretized_scores = upper_edges[np.searchsorted(upper_edges, scores, side='left')]
    less = np.searchsorted(np.sort(discretized_scores), upper_edges, side='left')
    greater = len(scores) - np.searchsorted(np.sort(discretized_scores), upper_edges, side='right')

    if 0 < q < 1:
        sensitivity = max(1 / q, 1 / (1 - q))
        weight = np.maximum(less / q, greater / (1 - q))
        weights = np.exp(-eps * weight / (2 * sensitivity))
    elif q >= 1:
        # Limit of the exponential mechanism as q approaches 1 from below.
        weights = np.exp(-eps * greater / 2)
    else:
        raise ValueError(f'q must be positive, got {q:.6g}')

    probabilities = weights / np.sum(weights)
    
    return float(rng.choice(upper_edges, p=probabilities))


# -------------------------------------------------------------------------
# 3. Algorithm 3: FedCP²-QQ, including Eq. (10) and gamma grid search
# -------------------------------------------------------------------------
def correction(alpha: float, gamma: float, eps: float, B: int, m: int) -> int:
    """
    Calculation of Equation (10) in the paper.
    """
    return ceil(2 / eps * np.log(B / (1 - (1 - gamma * alpha) ** (1 / m))))


def select_gamma(M: np.ndarray, alpha: float, eps: float, B: int) -> dict:
    """
    Finds the gamma that minimizes the coverage objective described in Section 4.
    
    When the corrected quantile order exceeds n, we cap it at n when evaluating
    the coverage objective. This workaround is not specified in the paper.
    """
    n, m = M.shape
    best = None

    for gamma in np.linspace(0.01, 0.99, 99):
        target = (1 - alpha) / (1 - gamma * alpha)

        try:
            ell, k = find_l_star_k_star(M, target)
        except ValueError:
            continue

        cor = correction(alpha, gamma, eps, B, m)
        order = ell + cor

        # Cap the order at n when evaluating the coverage objective.
        coverage = float(M[min(order, n) - 1, k - 1])

        if best is None or coverage < best['coverage']:
            best = {
                'gamma': float(gamma),
                'ell': ell,
                'k': k,
                'ell_cor': cor,
                'q': max(order / n, 0.5),
                'coverage': coverage
            }

    if best is None:
        raise ValueError(f'No candidate gamma for epsilon={eps}')

    return best


def fedcp2_qq(local_scores: np.ndarray, eps: float, bin_edges: np.ndarray,
              config: dict, rng: np.random.Generator) -> float:
    """
    Returns the global differentially private quantile-of-quantiles.
    """
    local_scores = np.asarray(local_scores, dtype=float)

    m, n = local_scores.shape

    client_quantiles = np.array([
        dp_quantile(local_scores[j], config['q'], eps, bin_edges, rng)
        for j in range(m)
    ])

    server_quantile = np.partition(client_quantiles, config['k'] - 1)[config['k'] - 1]

    return float(server_quantile)

# -------------------------------------------------------------------------
# 4. Appendix B.2 / Figure 18: experiment, OpenML loading, and box plots
# -------------------------------------------------------------------------
def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Loads Bike from the original Romano et al. CQR benchmark dataset,
    or Bio from OpenML (unchanged).
    """
    if name == 'bike':
        url = ('https://raw.githubusercontent.com/yromano/cqr/master/'
               'datasets/bike_train.csv')
        df = pd.read_csv(url)

        # Match the feature preparation in yromano/cqr/datasets/datasets.py.
        season = pd.get_dummies(df['season'], prefix='season', dtype=float)
        weather = pd.get_dummies(df['weather'], prefix='weather', dtype=float)
        df = pd.concat([df, season, weather], axis=1)
        df = df.drop(columns=['season', 'weather'])

        dates = pd.to_datetime(df['datetime'])
        df['hour'] = dates.dt.hour
        df['day'] = dates.dt.dayofweek
        df['month'] = dates.dt.month
        df['year'] = dates.dt.year.map({2011: 0, 2012: 1})

        df = df.drop(columns=['datetime', 'casual', 'registered'])
        X = df.drop(columns=['count']).astype(np.float32)
        y = df['count'].to_numpy(dtype=np.float32)
        return X, y

    elif name == 'bio':
        ds = fetch_openml(data_id=42903, target_column='RMSD', as_frame=True)
    else:
        raise ValueError('Dataset must be bike or bio')

    X = ds.data.copy()
    y = ds.target.to_numpy(dtype=float)

    return X, y


def preprocess(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame):
    """
    Prepares the features for the Quantile Random Forest.
    """
    train = pd.get_dummies(train, dtype=float)
    cal = pd.get_dummies(cal, dtype=float).reindex(columns=train.columns, fill_value=0)
    test = pd.get_dummies(test, dtype=float).reindex(columns=train.columns, fill_value=0)

    scaler = StandardScaler().fit(train)

    return scaler.transform(train), scaler.transform(cal), scaler.transform(test)


def cqr_scores(X_train, y_train, X_cal, y_cal, X_test, y_test,
               alpha: float, trees: int, seed: int):
    """
    Trains a Quantile Random Forest and calculates calibration and test CQR scores.
    """
    quantiles = [alpha / 2, 1 - alpha / 2]

    reg = RandomForestQuantileRegressor(
        n_estimators=trees,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed
    )
    reg.fit(X_train, y_train)

    cal_predictions = reg.predict(X_cal, quantiles=quantiles)
    test_predictions = reg.predict(X_test, quantiles=quantiles)

    cal_scores = np.maximum(cal_predictions[:, 0] - y_cal, y_cal - cal_predictions[:, 1])
    test_scores = np.maximum(test_predictions[:, 0] - y_test, y_test - test_predictions[:, 1])

    return cal_scores, test_scores


def run_experiment(dataset_name: str, n_splits: int, seed: int, *, m: int, n: int,
                   B: int, alpha: float,epsilons: tuple[float, ...], trees: int):
    """
    Runs repeated experiments comparing the empirical coverage of
    FedCP-QQ and FedCP²-QQ for different privacy levels.
    """

    X, y = load_dataset(dataset_name)

    # Calculate the quantile orders for both algorithms.
    M = calc_matrix_M(m, n, 0.5).T
    ell, k = find_l_star_k_star(M, 1 - alpha)
    configs = {eps: select_gamma(M, alpha, eps, B) for eps in epsilons}

    results = {'FedCP-QQ': [], **{f'ε = {eps:g}': [] for eps in epsilons}}

    for split in range(n_splits):
        split_seed = seed + split

        # Split the dataset into 40% training, 40% calibration, 20% test.
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.6, random_state=split_seed)
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_tmp, y_tmp, test_size=1/3, random_state=split_seed)

        # Select m clients with n calibration observations each.
        rng = np.random.default_rng(seed + 100_000 + split)  # Offset to use a separate random sequence
        chosen = rng.choice(len(y_cal), size=m * n, replace=False)
        X_cal, y_cal = X_cal.iloc[chosen], y_cal[chosen]

        # Train the forest and calculate CQR scores.
        train_features, cal_features, test_features = preprocess(X_train, X_cal, X_test)
        cal_scores, test_scores = cqr_scores(
            train_features, y_train,
            cal_features, y_cal,
            test_features, y_test,
            alpha, trees, split_seed
        )

        client_scores = cal_scores.reshape(m, n)

        # Algorithm 1: Non-private FedCP-QQ.
        threshold = fedcp_qq(client_scores, ell, k)
        results['FedCP-QQ'].append(np.mean(test_scores <= threshold))

        # Algorithm 3: Private FedCP²-QQ.
        # Include negative CQR scores in the bin range.
        edges = np.linspace(np.min(cal_scores), np.max(cal_scores), B + 1)

        for eps in epsilons:
            threshold = fedcp2_qq(client_scores, eps, edges, configs[eps], rng)
            results[f'ε = {eps:g}'].append(np.mean(test_scores <= threshold))

        print(f'{dataset_name}: split {split + 1}/{n_splits}')

    return results

def plot_results(results: dict, alpha: float, seed: int = 42):
    """
    Plots empirical coverage across dataset splits for each method.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(7.5 * len(results), 4.5))
    axes = np.atleast_1d(axes)

    rng = np.random.default_rng(seed + 9_999)
    colors = ['#8da0cb', '#b3cde3', '#decbe4', '#fed9a6']

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
                s=15, alpha=0.55, color='#555555'
            )
            ax.scatter(
                np.mean(coverages), position,
                s=58, facecolor='white', edgecolor='black', zorder=4
            )

        ax.axvline(1 - alpha, linestyle='--', color='black')
        ax.set_yticks(positions, labels)
        ax.set_xlabel('Coverage')
        ax.set_xlim(max(0, 1 - alpha - 0.06), 1.005)
        ax.set_title(dataset)
        ax.grid(axis='x', alpha=0.25)

    plt.tight_layout()
    plt.show()


def main():
    """
    Runs the experiments and displays the results.
    """
    names = ('bio', 'bike') if DATASET == 'both' else (DATASET,)

    all_results = {}
    for name in names:
        all_results[name] = run_experiment(
            name, SPLITS, SEED,
            m=M, n=N, B=B, alpha=ALPHA,
            epsilons=EPSILONS, trees=TREES
        )

    plot_results(all_results, ALPHA, SEED)

    for name, values in all_results.items():
        for method, coverages in values.items():
            print(f'{name:<5} {method:<13}: {np.mean(coverages):.4f} ± {np.std(coverages):.4f}')

    return all_results


if __name__ == '__main__':
    results = main()