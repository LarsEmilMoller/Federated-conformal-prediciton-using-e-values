#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:56:25 2026

@author: livboethrysoe
"""

import matplotlib.pyplot as plt
import numpy as np
from math import ceil, log
from scipy.stats import binom
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# 1. Exact Coverage Matrix M_{l, k} via Appendix A.2 (Lebrun, 2013)
# -------------------------------------------------------------------------
def compute_M_matrix(m: int, n: int, t: float = 0.5) -> np.ndarray:
    M = np.zeros((n, m))
    x = np.arange(n + 1)
    pmf_W = binom.pmf(x, n, t)
    pmf_sum_W = binom.pmf(np.arange(m * n + 1), m * n, t)

    for l in range(1, n + 1):
        prob_hi = np.sum(pmf_W[l:])
        pmf_T_hi = np.zeros(n + 1)
        if prob_hi > 0:
            pmf_T_hi[l:] = pmf_W[l:] / prob_hi

        prob_lo = np.sum(pmf_W[:l])
        pmf_T_lo = np.zeros(n + 1)
        if prob_lo > 0:
            pmf_T_lo[:l] = pmf_W[:l] / prob_lo

        hi_powers = [np.array([1.0])]
        for _ in range(m):
            hi_powers.append(np.convolve(hi_powers[-1], pmf_T_hi))

        lo_powers = [np.array([1.0])]
        for _ in range(m):
            lo_powers.append(np.convolve(lo_powers[-1], pmf_T_lo))

        term_j = np.zeros(m + 1)
        for j in range(m + 1):
            if (j > 0 and prob_hi == 0) or (m - j > 0 and prob_lo == 0):
                continue
            pmf_sum_T = np.convolve(hi_powers[j], lo_powers[m - j])
            r_vals = np.arange(len(pmf_sum_T))
            valid = (pmf_sum_T > 0) & (pmf_sum_W[r_vals] > 0)

            p_r = (
                pmf_sum_T[valid]
                * (prob_hi**j)
                * (prob_lo ** (m - j))
                / pmf_sum_W[r_vals[valid]]
            )
            term_j[j] = binom.pmf(j, m, 0.5) * (2**m) * np.sum(p_r)

        for k in range(1, m + 1):
            M[l - 1, k - 1] = 1.0 - np.sum(term_j[k:]) / (m * n + 1)

    return M


def find_l_star_k_star(M: np.ndarray, target_cov: float) -> tuple[int, int]:
    valid_indices = np.argwhere(M >= target_cov)
    if len(valid_indices) == 0:
        return M.shape[0], M.shape[1]
    valid_values = M[valid_indices[:, 0], valid_indices[:, 1]]
    best_idx = valid_indices[np.argmin(valid_values)]
    return int(best_idx[0] + 1), int(best_idx[1] + 1)


# -------------------------------------------------------------------------
# 2. Algorithm 2: Differentially Private Quantile (Exponential Mechanism)
# -------------------------------------------------------------------------
def dp_quantile(scores: np.ndarray, q: float, eps: float, bin_edges: np.ndarray, rng: np.random.Generator) -> float:
    B = len(bin_edges) - 1
    e_b = bin_edges[1:]
    bin_indices = np.clip(np.searchsorted(e_b, scores, side="left"), 0, B - 1)
    S_bar = e_b[bin_indices]

    less_counts = np.array([np.sum(S_bar < eb) for eb in e_b])
    greater_counts = np.array([np.sum(S_bar > eb) for eb in e_b])

    w_b = np.maximum(less_counts / q, greater_counts / (1.0 - q))
    delta_q = max(1.0 / q, 1.0 / (1.0 - q))

    log_probs = -(eps * w_b) / (2.0 * delta_q)
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    probs /= np.sum(probs)

    chosen_idx = rng.choice(B, p=probs)
    return float(e_b[chosen_idx])


# -------------------------------------------------------------------------
# 3. Algorithm 3: FedCP^2-QQ
# -------------------------------------------------------------------------
def select_gamma(M: np.ndarray, alpha: float, eps: float, B: int, m: int, n: int) -> float:
    gammas = np.linspace(0.01, 0.99, 99)
    best_gamma = 0.5
    best_cov = np.inf

    for gamma in gammas:
        target_cov = (1.0 - alpha) / (1.0 - gamma * alpha)
        if target_cov >= 1.0:
            continue
        l_g, k_g = find_l_star_k_star(M, target_cov)
        l_cor = ceil((2.0 / eps) * log(B / (1.0 - (1.0 - gamma * alpha) ** (1.0 / m))))
        l_total = min(l_g + l_cor, n)
        cov_val = M[l_total - 1, k_g - 1]
        if cov_val < best_cov:
            best_cov = cov_val
            best_gamma = gamma

    return float(best_gamma)


def fedcp2_qq(
    local_scores: list[np.ndarray],
    alpha: float,
    M: np.ndarray,
    eps: float,
    bin_edges: np.ndarray,
    gamma: float | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()

    m = len(local_scores)
    n = len(local_scores[0])
    B = len(bin_edges) - 1

    if gamma is None:
        gamma = select_gamma(M, alpha, eps, B, m, n)

    target_cov = (1.0 - alpha) / (1.0 - gamma * alpha)
    l_gamma, k_gamma = find_l_star_k_star(M, target_cov)

    l_cor = ceil((2.0 / eps) * log(B / (1.0 - (1.0 - gamma * alpha) ** (1.0 / m))))
    q = max((l_gamma + l_cor) / n, 0.5)
    q = min(q, 1.0 - 1e-9)

    private_quantiles = [
        dp_quantile(local_scores[j], q=q, eps=eps, bin_edges=bin_edges, rng=rng)
        for j in range(m)
    ]

    sorted_q = np.sort(private_quantiles)
    return float(sorted_q[k_gamma - 1])


# -------------------------------------------------------------------------
# 4. Appendix B.2 Benchmark Runner & Publication-Ready Plotting
# -------------------------------------------------------------------------
def run_appendix_b2_experiment(dataset_name: str = "bike", n_splits: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)

    if dataset_name == "bike":
        data = fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True, parser="auto")
        df = data.frame if hasattr(data, "frame") and data.frame is not None else data
        df = df.dropna()
        
        target_col = data.details.get("default_target_attribute", df.columns[-1]) if hasattr(data, "details") else df.columns[-1]
        if target_col not in df.columns:
            target_col = df.columns[-1]
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
            
        X = df[numeric_cols].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)
        
    elif dataset_name == "bio":
        data = fetch_openml(data_id=42903, as_frame=False, parser="auto")
        X = np.nan_to_num(data.data.astype(float))
        y = data.target.astype(float)
    else:
        raise ValueError("Choose 'bike' or 'bio'.")

    m, n = 5, 200
    B = 100
    alpha = 0.1
    epsilons = [10.0, 5.0, 1.0]

    print(f"Precomputing M matrix for m={m}, n={n}...")
    M = compute_M_matrix(m, n)
    l_star, k_star = find_l_star_k_star(M, 1.0 - alpha)

    results = {
        "FedCP-QQ": [],
        "$\\varepsilon = 10$": [],
        "$\\varepsilon = 5$": [],
        "$\\varepsilon = 1$": []
    }

    for split in range(n_splits):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=seed + split)
        X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=seed + split)

        cal_idx = rng.choice(len(X_cal), size=m * n, replace=False)
        X_cal, y_cal = X_cal[cal_idx], y_cal[cal_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_cal = scaler.transform(X_cal)
        X_test = scaler.transform(X_test)

        q_lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, n_estimators=100, random_state=seed + split)
        q_hi = GradientBoostingRegressor(loss="quantile", alpha=1.0 - alpha / 2, n_estimators=100, random_state=seed + split)
        q_lo.fit(X_train, y_train)
        q_hi.fit(X_train, y_train)

        cal_scores = np.maximum(q_lo.predict(X_cal) - y_cal, y_cal - q_hi.predict(X_cal))
        test_scores = np.maximum(q_lo.predict(X_test) - y_test, y_test - q_hi.predict(X_test))

        local_scores = [cal_scores[j * n : (j + 1) * n] for j in range(m)]

        # Non-private baseline
        local_q = [np.sort(s)[l_star - 1] for s in local_scores]
        q_hat_nonpriv = np.sort(local_q)[k_star - 1]
        results["FedCP-QQ"].append(np.mean(test_scores <= q_hat_nonpriv))

        # Private FedCP^2-QQ across epsilons
        S_min = min(0.0, float(np.min(cal_scores)))
        S_max = float(np.max(cal_scores))
        bin_edges = np.linspace(S_min, S_max, B + 1)

        priv_keys = ["$\\varepsilon = 10$", "$\\varepsilon = 5$", "$\\varepsilon = 1$"]
        for eps, key in zip(epsilons, priv_keys):
            q_hat_priv = fedcp2_qq(local_scores, alpha=alpha, M=M, eps=eps, bin_edges=bin_edges, rng=rng)
            results[key].append(np.mean(test_scores <= q_hat_priv))

    print(f"\nResults on '{dataset_name}' over {n_splits} splits (Target Coverage = {1 - alpha:.2f}):")
    for method, covs in results.items():
        print(f"  {method:<20} | Mean: {np.mean(covs):.4f} (std: {np.std(covs):.4f})")

    # Plotting results
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = list(results.keys())
    data_to_plot = list(results.values())
    y_positions = np.arange(len(labels), 0, -1)

    colors = ['#8da0cb', '#b3cde3', '#decbe4', '#fed9a6']
    
    bp = ax.boxplot(data_to_plot, positions=y_positions, vert=False, patch_artist=True,
                    widths=0.4, showcaps=True, showfliers=True,
                    whiskerprops=dict(color='#4d4d4d', linewidth=1.5),
                    capprops=dict(color='#4d4d4d', linewidth=1.5),
                    medianprops=dict(color='black', linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#4d4d4d')
        patch.set_linewidth(1.5)

    for i, (key, values) in enumerate(results.items()):
        y_pos = y_positions[i]
        vals = np.array(values)
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(vals, y_pos + jitter, color='#4d4d4d', alpha=0.7, s=15, zorder=3)
        mean_val = np.mean(vals)
        ax.scatter([mean_val], [y_pos], color='white', edgecolor='black', s=50, zorder=4, marker='o')

    ax.axvline(x=1.0 - alpha, color='black', linestyle='--', linewidth=1.5, zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_xlim(0.85, 1.00)
    ax.set_xlabel("coverage", fontsize=14)
    ax.set_title(dataset_name, fontsize=18, pad=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4d4d4d')
    ax.spines['bottom'].set_color('#4d4d4d')
    ax.xaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.yaxis.grid(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change dataset_name to "bike" or "bio" as needed
    run_appendix_b2_experiment(dataset_name="bike", n_splits=20)