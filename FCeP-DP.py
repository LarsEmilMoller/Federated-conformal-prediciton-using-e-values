#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:21:09 2026

@author: livboethrysoe
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# Exact Algorithm 4: FCeP-private Implementation
# -------------------------------------------------------------------------
def fcep_private_eval(
    local_scores: list[np.ndarray],
    X_test_single: np.ndarray,
    y_test_single: float,
    alpha: float,
    eps: float,
    S_max: float,
    q_lo_model,
    q_hi_model,
    rng: np.random.Generator
) -> bool:
    """
    Direct implementation of Algorithm 4 (FCeP-private).
    """
    K = len(local_scores)
    n_clients = [len(s) for s in local_scores]
    N = sum(n_clients)

    # --- CLIENTS PHASE ---
    hat_S_k = []
    Z_k_list = []
    for k in range(K):
        S_k = np.sum(local_scores[k])
        # Draw local Laplace noise Z_k ~ Lap(0, S_max / epsilon)
        scale = S_max / eps
        Z_k = rng.laplace(0.0, scale)
        Z_k_list.append(Z_k)
        
        hat_S_k.append(S_k + Z_k)
        # Sends (hat_S_k, n_k) to server (modeled implicitly here)

    # --- SERVER PHASE ---
    hat_S_G = sum(hat_S_k)
    Z_G = sum(Z_k_list)

    # Compute conformal score S(X_test, y) for the test point and true label
    lo_pred = q_lo_model.predict(X_test_single.reshape(1, -1))[0]
    hi_pred = q_hi_model.predict(X_test_single.reshape(1, -1))[0]
    s_test = max(lo_pred - y_test_single, y_test_single - hi_pred)
    s_test = max(0.0, min(s_test, S_max)) # Bounded to [0, S_max]

    # Evaluate for the test label y
    # A(y) definition from algorithm
    term_A = (1.0 / (N + K)) * (hat_S_G - Z_G + s_test + (K - 1))
    
    if term_A <= 0:
        return False

    # Non-private federated e-value
    E_fed = s_test / term_A

    # Corrected privatized e-value formula
    term_corr_1 = hat_S_G / (hat_S_G + Z_G) if (hat_S_G + Z_G) != 0 else 1.0
    term_corr_2 = 1.0 / (1.0 + (2.0 * K * (S_max ** 2)) / ((eps ** 2) * (term_A ** 2)))
    
    E_priv = E_fed * term_corr_1 * term_corr_2

    # Acceptance condition: E_priv < 1 / alpha
    return bool(E_priv < (1.0 / alpha))


# -------------------------------------------------------------------------
# Benchmark & Box Plot Comparison Runner
# -------------------------------------------------------------------------
def run_fcep_private_benchmark(dataset_name: str = "bike", n_splits: int = 10, seed: int = 42):
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

    K, n = 5, 200
    alpha = 0.1
    epsilons = [10.0, 1.0]

    results = {
        "FCeP (Non-Private)": [],
        "FCeP-private ($\\varepsilon = 10$)": [],
        "FCeP-private ($\\varepsilon = 1$)": []
    }

    for split in range(n_splits):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=seed + split)
        X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=seed + split)

        cal_idx = rng.choice(len(X_cal), size=K * n, replace=False)
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
        S_max = float(np.max(cal_scores))
        local_scores = [cal_scores[k * n : (k + 1) * n] for k in range(K)]

        test_subset_idx = rng.choice(len(X_test), size=50, replace=False)
        X_t_sub = X_test[test_subset_idx]
        y_t_sub = y_test[test_subset_idx]

        # 1. Non-Private Baseline Evaluation
        S_G_raw = sum(np.sum(s) for s in local_scores)
        cov_non_priv = 0
        for i in range(len(X_t_sub)):
            lo_p = q_lo.predict(X_t_sub[i].reshape(1, -1))[0]
            hi_p = q_hi.predict(X_t_sub[i].reshape(1, -1))[0]
            s_test = max(0.0, min(max(lo_p - y_t_sub[i], y_t_sub[i] - hi_p), S_max))
            A_val = (1.0 / (K * n + K)) * (S_G_raw + s_test + (K - 1))
            e_val = s_test / A_val if A_val > 0 else 0.0
            if e_val < (1.0 / alpha):
                cov_non_priv += 1
        results["FCeP (Non-Private)"].append(cov_non_priv / len(X_t_sub))

        # 2. FCeP-private Across Privacy Budgets
        for eps in epsilons:
            cov_priv = 0
            for i in range(len(X_t_sub)):
                is_accepted = fcep_private_eval(
                    local_scores=local_scores,
                    X_test_single=X_t_sub[i],
                    y_test_single=y_t_sub[i],
                    alpha=alpha,
                    eps=eps,
                    S_max=S_max,
                    q_lo_model=q_lo,
                    q_hi_model=q_hi,
                    rng=rng
                )
                if is_accepted:
                    cov_priv += 1
            key_name = f"FCeP-private ($\\varepsilon = {int(eps)}$)"
            results[key_name].append(cov_priv / len(X_t_sub))

    # -------------------------------------------------------------------------
    # Publication-Ready Plotting
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.5))
    labels = list(results.keys())
    data_to_plot = list(results.values())
    y_positions = np.arange(len(labels), 0, -1)

    colors = ['#8da0cb', '#b3cde3', '#fed9a6']
    
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
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_xlim(0.80, 1.00)
    ax.set_xlabel("coverage", fontsize=13)
    ax.set_title(f"FCeP-private Evaluation on '{dataset_name}'", fontsize=15, pad=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4d4d4d')
    ax.spines['bottom'].set_color('#4d4d4d')
    ax.xaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.yaxis.grid(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_fcep_private_benchmark(dataset_name="bike", n_splits=10)