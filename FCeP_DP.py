#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:24:17 2026

@author: livboethrysoe
"""

"""
FCeP-DP — locally differentially private Federated Conformal e-Prediction.

Implements the algorithm supplied by the user (client-side sums with truncated
Laplace noise, followed by the specified server-side e-value). The prediction
model is Ridge with 5-fold GridSearchCV and the score is bounded absolute residual split CP:
    S(x, y) = min(|y - f_hat(x)|, S_MAX).

Experiment layout matches fedcp2_qq_ridge_skip_infeasible.py:
- Bike / Bio benchmark loaders, preprocessing, 40/40/20 splits
- Same random selection of K * N calibration observations
- Ridge with 5-fold GridSearchCV on training data only, 20 splits, K=5, N=200, seed=42
- Coverage-only horizontal boxplots, original pastel palette

IMPORTANT:
  * S_MAX must be fixed independently of PRIVATE calibration data. The
    provided per-dataset defaults are illustrative public constants, not
    empirically calibrated guarantees of narrow intervals. Explicitly
    truncating absolute residuals makes the score bounded as required.
  * With a capped score, the prediction set need not be a finite interval;
    the exact membership test here is E_y < 1/alpha, as in the algorithm.
  * There is NO claim here that the displayed e-value formula has been proved
    to have the desired statistical coverage. This script implements it as
    supplied, without replacing it with another correction.
  * This private protocol releases one noisy sum per client, not per-test
    outputs. A pre-trained model and fixed public S_MAX are assumed.
  * To compare the two algorithms rigorously, use the SAME bounded score,
    train/cal/test splits, model, and privacy accounting in both scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


# ========================================================================
# EXPERIMENT SETTINGS — change these values, then rerun the script
# ========================================================================
DATASET = 'bike'             # 'bike', 'bio', or 'both'
ALPHA = 0.3
EPSILONS = (20.0, 10.0, 5.0, 1.0, 0.5, 0.05)
DELTA = 1e-4                # Per-client (epsilon, delta)-DP parameters
K = 5                       # Clients (the other script names this M)
N = 200                     # Calibration observations per client
SPLITS = 20
SEED = 42

RIDGE_ALPHAS = np.logspace(-3, 3, 13)
RIDGE_CV_FOLDS = 5

# Fixed public score caps. These are EXAMPLES, not estimated from calibration
# scores. Adjust based on domain knowledge before interpreting results.
S_MAX_BY_DATASET = {'bike': 1000.0, 'bio': 100.0}
# ========================================================================


# -------------------------------------------------------------------------
# 1. Client mechanism: bounded score sums + truncated Laplace noise
# -------------------------------------------------------------------------
def bounded_score(y_true, y_pred, s_max: float) -> np.ndarray:
    """Nonconformity score S(x,y) = min(|y - f_hat(x)|, S_MAX)."""
    if not np.isfinite(s_max) or s_max <= 0:
        raise ValueError('S_MAX must be a positive, finite public constant')
    scores = np.minimum(np.abs(np.asarray(y_true) - np.asarray(y_pred)), s_max)
    if not np.all(np.isfinite(scores)):
        raise ValueError('Non-finite conformity scores')
    return scores


def truncation_bound(s_max: float, eps: float, delta: float) -> float:
    """T = (S_MAX/eps) log[1 + (exp(eps)-1)/(2 delta)]."""
    if not (np.isfinite(s_max) and s_max > 0 and np.isfinite(eps) and eps > 0):
        raise ValueError('S_MAX and epsilon must be positive and finite')
    if not (np.isfinite(delta) and 0 < delta < 0.5):
        raise ValueError('delta must be in (0, 1/2)')
    # Avoid exp(eps) overflow; logaddexp also behaves well for small eps.
    # log(1 + (exp(eps)-1)/(2delta))
    log_ratio = np.log(np.expm1(eps)) - np.log(2.0 * delta) if eps < 700 else (
        eps + np.log1p(-np.exp(-eps)) - np.log(2.0 * delta)
    )
    return float((s_max / eps) * np.logaddexp(0.0, log_ratio))


def sample_truncated_laplace(
    rng: np.random.Generator, s_max: float, eps: float, delta: float,
    size=None
) -> np.ndarray:
    """Draw from Laplace(0, b) conditioned on |Z| <= T (no boundary atoms).

    Conditional magnitude CDF: P(|Z| <= r | |Z| <= T)
                          = (1 - exp(-r/b))/(1 - exp(-T/b)).
    """
    b = s_max / eps
    T = truncation_bound(s_max, eps, delta)
    u = rng.random(size=size)
    trunc_mass = -np.expm1(-T / b)                 # 1 - exp(-T/b)
    magnitude = -b * np.log1p(-u * trunc_mass)
    sign = np.where(rng.random(size=size) < 0.5, -1.0, 1.0)
    return sign * magnitude


def client_noisy_sums(client_scores: np.ndarray, eps: float, delta: float,
                      s_max: float, rng: np.random.Generator):
    """Each client sends (sum(scores) + TLap noise, n_k)."""
    client_scores = np.asarray(client_scores, dtype=float)
    if client_scores.ndim != 2 or client_scores.shape[1] == 0:
        raise ValueError('Expected 2D matrix of equal-sized client calibration sets')
    if not np.all(np.isfinite(client_scores)) or np.any(client_scores < 0) or np.any(client_scores > s_max):
        raise ValueError('Client scores must lie in [0, S_MAX]')
    k, n = client_scores.shape
    sums = client_scores.sum(axis=1)
    noises = sample_truncated_laplace(rng, s_max, eps, delta, size=k)
    noisy_sums = sums + noises
    counts = np.full(k, n, dtype=int)
    return noisy_sums, counts


# -------------------------------------------------------------------------
# 2. Server: e-values and the prediction-set membership test
# -------------------------------------------------------------------------
def private_e_values(test_scores: np.ndarray, noisy_sums: np.ndarray,
                     counts: np.ndarray, eps: float, delta: float,
                     s_max: float, negative_stats=None) -> np.ndarray:
    """Compute EXACTLY the e-value expression supplied in the algorithm.

    E_y = (N_total+K)*S_y/A_y
          - 2*K*S_MAX**2 / [eps**2*(A_y-K*T)**2],
    A_y = sum(noisy client sums) + S_y + (K-1)*S_MAX.

    Fails explicitly if a denominator is numerically zero. No heuristic
    capping, denominator clipping, or modification to the formula.
    """
    scores = np.asarray(test_scores, dtype=float)
    noisy_sums = np.asarray(noisy_sums, dtype=float)
    counts = np.asarray(counts, dtype=int)
    if noisy_sums.ndim != 1 or counts.ndim != 1 or noisy_sums.size != counts.size:
        raise ValueError('One noisy sum and count required per client')
    if scores.ndim != 1 or np.any(scores < 0) or np.any(scores > s_max):
        raise ValueError('Test scores must be 1D and bounded by S_MAX')
    k = len(counts)
    n_total = int(counts.sum())
    T = truncation_bound(s_max, eps, delta)
    A = noisy_sums.sum() + scores + (k - 1) * s_max
    lower_A = A - k * T
    if np.any(~np.isfinite(A)) or np.any(~np.isfinite(lower_A)):
        raise FloatingPointError('Non-finite denominator in e-value expression')
    # We do not impose A-K*T>0, since the supplied expression only squares it.
    # But zero denominators would make E undefined.
    if np.any(np.isclose(A, 0.0, atol=1e-12, rtol=0)) or np.any(np.isclose(lower_A, 0.0, atol=1e-12, rtol=0)):
        raise FloatingPointError('E-value denominator equals (or is too close to) zero')
    values = ((n_total + k) * scores / A
              - 2.0 * k * s_max**2 / (eps**2 * lower_A**2))
    if not np.all(np.isfinite(values)):
        raise FloatingPointError('Non-finite e-value')
    negative_count = int(np.count_nonzero(values < 0))
    if negative_stats is not None:
        negative_stats[0] += negative_count
        negative_stats[1] += values.size
    values = np.maximum(0, values)
    return values


def prediction_set_membership(x_test, y_candidates, model, noisy_sums,
                              counts, alpha: float, eps: float,
                              delta: float, s_max: float) -> np.ndarray:
    """For one test feature vector, test arbitrary candidate responses y.

    Returns a Boolean mask defining C_N(x_test) = {y: E_y < 1/alpha}.
    A capped score can make this set non-interval-shaped, so we do not
    assume it can be represented by a single pair of endpoints.
    """
    y_candidates = np.asarray(y_candidates, dtype=float)
    if y_candidates.ndim != 1:
        raise ValueError('y_candidates must be a 1D array')
    x_test = np.asarray(x_test, dtype=float).reshape(1, -1)
    prediction = float(model.predict(x_test)[0])
    candidate_scores = bounded_score(y_candidates, prediction, s_max)
    e_values = private_e_values(candidate_scores, noisy_sums, counts,
                                eps, delta, s_max)
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0, 1)')
    return e_values < 1.0 / alpha


def evaluate_coverage(test_scores: np.ndarray, noisy_sums: np.ndarray,
                      counts: np.ndarray, alpha: float, eps: float,
                      delta: float, s_max: float, negative_stats=None) -> float:
    """Fraction of test responses for which E_y < 1/alpha (strictly)."""
    if not (0 < alpha < 1):
        raise ValueError('alpha must lie in (0, 1)')
    e_values = private_e_values(test_scores, noisy_sums, counts, eps, delta, s_max,
                                negative_stats=negative_stats)
    return float(np.mean(e_values < 1.0 / alpha))


# -------------------------------------------------------------------------
# 3. Datasets and Ridge split CP (same as the reference experiment)
# -------------------------------------------------------------------------
def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Loads the same Bike / Bio sources and features as the FedCP²-QQ script."""
    if name == 'bike':
        url = ('https://raw.githubusercontent.com/yromano/cqr/master/'
               'datasets/bike_train.csv')
        df = pd.read_csv(url)
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
    """Prepares features for Ridge regression, fitted using training only."""
    train = pd.get_dummies(train, dtype=float)
    cal = pd.get_dummies(cal, dtype=float).reindex(columns=train.columns, fill_value=0)
    test = pd.get_dummies(test, dtype=float).reindex(columns=train.columns, fill_value=0)

    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(cal), scaler.transform(test)


def ridge_split_cp_scores(X_train, y_train, X_cal, y_cal, X_test, y_test):
    """Ridge with SVD solver, selected by the same 5-fold MSE CV grid."""
    reg = GridSearchCV(
        estimator=Ridge(solver='svd'),
        param_grid={'alpha': RIDGE_ALPHAS},
        cv=RIDGE_CV_FOLDS,
        scoring='neg_mean_squared_error'
    )
    reg.fit(X_train, y_train)
    print(f"Selected Ridge alpha: {reg.best_params_['alpha']:.6g}")

    cal_scores = np.abs(y_cal - reg.predict(X_cal))
    test_scores = np.abs(y_test - reg.predict(X_test))
    return cal_scores, test_scores


# -------------------------------------------------------------------------
# 4. Repeated experiment and original coverage-only boxplots
# -------------------------------------------------------------------------
def run_experiment(dataset_name: str, n_splits: int, seed: int, *,
                   k: int, n: int, alpha: float,
                   epsilons: tuple[float, ...], delta: float):
    X, y = load_dataset(dataset_name)
    s_max = S_MAX_BY_DATASET[dataset_name]
    results = {f'ε = {eps:g}': [] for eps in epsilons}
    negative_counts = {eps: [0, 0] for eps in epsilons}

    for split in range(n_splits):
        split_seed = seed + split
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.6, random_state=split_seed)
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_tmp, y_tmp, test_size=1/3, random_state=split_seed)

        if len(y_cal) < k * n:
            raise ValueError(f'Not enough calibration samples for {k} clients of size {n}')
        rng = np.random.default_rng(seed + 100_000 + split)
        chosen = rng.choice(len(y_cal), size=k * n, replace=False)
        X_cal, y_cal = X_cal.iloc[chosen], y_cal[chosen]

        train_features, cal_features, test_features = preprocess(X_train, X_cal, X_test)
        cal_scores, test_scores = ridge_split_cp_scores(
            train_features, y_train, cal_features, y_cal, test_features, y_test)

        # Required by FCeP-DP: score values must be bounded in [0, S_MAX].
        # The same score transformation is applied to calibration and test.
        cal_scores = np.minimum(cal_scores, s_max)
        test_scores = np.minimum(test_scores, s_max)
        clients = cal_scores.reshape(k, n)

        for eps in epsilons:
            # Independent privacy randomness per epsilon, but same data split.
            privacy_rng = np.random.default_rng(seed + 1_000_000 + 1000 * split +
                                                int(round(eps * 100)))
            noisy_sums, counts = client_noisy_sums(clients, eps, delta, s_max, privacy_rng)
            results[f'ε = {eps:g}'].append(
                evaluate_coverage(test_scores, noisy_sums, counts,
                                  alpha, eps, delta, s_max,
                                  negative_stats=negative_counts[eps]))

        print(f'{dataset_name}: split {split + 1}/{n_splits}')
    return results, negative_counts


def plot_results(results: dict, alpha: float, delta: float, seed: int = 42):
    """Original pastel horizontal boxplots: COVERAGE is the x-axis."""
    fig, axes = plt.subplots(1, len(results), figsize=(7.5 * len(results), 4.5))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(seed + 9_999)
    colors = ['#8da0cb', '#b3cde3', '#decbe4', '#fed9a6']

    for ax, (dataset, values) in zip(axes, results.items()):
        labels = list(values)
        positions = np.arange(len(labels), 0, -1)
        boxes = ax.boxplot(
            list(values.values()), positions=positions, vert=False,
            patch_artist=True, widths=0.4, whis=(0, 100), showfliers=False)
        for box, color in zip(boxes['boxes'], colors):
            box.set_facecolor(color)
        for position, coverages in zip(positions, values.values()):
            ax.scatter(coverages,
                       position + rng.normal(0, 0.04, len(coverages)),
                       s=15, alpha=0.55, color='#555555')
            ax.scatter(np.mean(coverages), position,
                       s=58, facecolor='white', edgecolor='black', zorder=4)

        ax.axvline(1 - alpha, linestyle='--', color='black')
        ax.set_yticks(positions, labels)
        ax.set_xlabel('Coverage')
        ax.set_xlim(max(0, 1 - alpha - 0.06), 1.005)
        ax.set_title(f"{dataset} (α={alpha:g}, δ={delta:g})")
        ax.grid(axis='x', alpha=0.25)

    plt.tight_layout()
    plt.show()


def main():
    names = ('bio', 'bike') if DATASET == 'both' else (DATASET,)
    all_results = {}
    all_negative_counts = {}
    for name in names:
        all_results[name], all_negative_counts[name] = run_experiment(
            name, SPLITS, SEED, k=K, n=N, alpha=ALPHA,
            epsilons=EPSILONS, delta=DELTA)
    plot_results(all_results, ALPHA, DELTA, SEED)
    for name, values in all_results.items():
        for method, coverages in values.items():
            print(f'{name:<5} {method:<13}: '
                  f'{np.mean(coverages):.4f} ± {np.std(coverages):.4f}')
    print('\nNegative e-values (before clipping):')
    for name in names:
        for eps in EPSILONS:
            negative, total = all_negative_counts[name][eps]
            print(f'{name}: epsilon={eps:g}, delta={DELTA:g}: '
                  f'{negative} out of {total} e-values negative')
    return all_results


if __name__ == '__main__':
    results = main()





