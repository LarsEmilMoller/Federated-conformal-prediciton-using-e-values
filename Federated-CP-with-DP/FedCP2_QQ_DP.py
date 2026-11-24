#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:51:56 2026

@author: livboethrysoe
"""

"""
Implementation of FedCP-QQ and FedCP²-QQ following Algorithms 1–3
in "One-Shot Federated Conformal Prediction" by Humbert et al. (2023).

Includes computation of the coverage matrix M, selection of quantile
orders and computation of private quantiles using the exponential mechanism.

Reference implementation: https://github.com/pierreHmbt/FedCP-QQ
"""

from math import ceil
import numpy as np
from scipy.stats import binom
import scipy.special as sc


# -------------------------------------------------------------------------
# 1. Compute coverage matrix
# -------------------------------------------------------------------------
"""
Computation of the coverage matrix M.

The functions sum_hypergeo and calc_matrix_M are taken from
class_CP_QQ.py in the authors' GitHub repository, with minor
modifications to improve numerical stability.

Following the authors' implementation, M has dimensions (m, n),
with rows corresponding to k and columns corresponding to ell.
"""

def sum_hypergeo(aa, bb, n, t, epsi=0):
    m = len(aa)

    rv1 = binom(n, t)
    prob = []
    for i in range(m):
        prob.append(np.sum(rv1.pmf(np.arange(aa[i], bb[i] + 1))))

    r1 = 1.
    for i in range(m):
        r1 *= prob[i]

    rv2 = binom(m*n, t)

    gen = []
    for i in range(m):
        tt = np.sum(rv1.pmf(np.arange(aa[i], bb[i] + 1)))

        pp = []
        for j in np.arange(n, -1, -1):
            if j <= bb[i] and j >= aa[i]:
                pp.append(rv1.pmf(j)/tt)
            else:
                pp.append(0)
        gen.append(np.poly1d(pp))

    poly = gen[0]
    for i in range(1, len(gen)):
        poly = np.polymul(poly, gen[i])

    s = 0
    for l in range(np.sum(aa), np.sum(bb) + 1):
        r2 = rv2.pmf(l)
        if r2 > epsi:
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

                s += sum_hypergeo(
                    a_range, b_range, n, .5, epsi
                )*sc.binom(m, j)

            A[aa, bb] = 1-s/(m*n+1)

    return A


# -------------------------------------------------------------------------
# 2. Selection of quantile orders
# -------------------------------------------------------------------------
def find_l_star_k_star(M: np.ndarray, target_cov: float) -> tuple[int, int]:
    """
    Return the quantile orders with the smallest M_{ell,k} >= target_cov.
    """

    # Small tolerance to avoid floating-point rounding errors.
    valid = np.argwhere(M >= target_cov - 1e-12)

    if len(valid) == 0:
        raise ValueError(
            f'No quantile orders achieve coverage {target_cov:.6f}'
        )

    k, ell = min(valid, key=lambda idx: M[tuple(idx)])

    return int(ell + 1), int(k + 1)


# -------------------------------------------------------------------------
# 3. Algorithm 1: FedCP-QQ
# -------------------------------------------------------------------------
def fedcp_qq(local_scores: np.ndarray, ell: int, k: int) -> float:
    """
    Return the global quantile-of-quantiles.
    """

    client_quantiles = np.partition(local_scores, ell - 1, axis=1)[:, ell - 1]
    server_quantile = np.partition(client_quantiles, k - 1)[k - 1]

    return float(server_quantile)


# -------------------------------------------------------------------------
# 4. Algorithm 2: Differentially private quantile
# -------------------------------------------------------------------------
def dp_quantile(scores: np.ndarray, q: float, eps: float,
                bin_edges: np.ndarray, rng: np.random.Generator) -> float:
    """
    Return a differentially private quantile using the exponential mechanism.

    Algorithm 2 is defined only for 0 < q < 1. However, q >= 1 may occur
    in Algorithm 3 when the privacy correction ell_cor is sufficiently large.
    Invalid quantile levels raise an error rather than using an approximation.
    """

    # Convert scores and bin edges to floating-point NumPy arrays.
    scores = np.asarray(scores, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)

    # Check validity of quantile level and score range.
    if not 0 < q < 1:
        raise ValueError(f'Algorithm 2 requires 0 < q < 1, got q={q:.6g}')

    if np.any(scores < edges[0]) or np.any(scores > edges[-1]):
        raise ValueError('Scores outside the chosen bin range')

    # Extract the upper endpoint of each bin.
    upper_edges = edges[1:]

    # Discretize each score to the upper endpoint of its corresponding bin.
    discretized_scores = upper_edges[
        np.searchsorted(upper_edges, scores, side='left')
    ]

    # Count the discretized scores strictly below and above each bin endpoint.
    less = np.searchsorted(
        np.sort(discretized_scores), upper_edges, side='left'
    )
    greater = len(scores) - np.searchsorted(
        np.sort(discretized_scores), upper_edges, side='right'
    )

    # Compute the bin weights and the sensitivity.
    weight = np.maximum(less / q, greater / (1 - q))
    sensitivity = max(1 / q, 1 / (1 - q))
    
    # Compute probabilities using numerically stable normalization.
    log_weights = -eps * weight / (2 * sensitivity)
    weights = np.exp(log_weights - np.max(log_weights))
    probabilities = weights / np.sum(weights)

    # Sample a bin endpoint according to the computed probabilities.
    return float(rng.choice(upper_edges, p=probabilities))


# -------------------------------------------------------------------------
# 5. Algorithm 3: FedCP²-QQ
# -------------------------------------------------------------------------
def correction(alpha: float, gamma: float, eps: float, B: int, m: int) -> int:
    """
    Compute the privacy correction ell_cor defined in Equation (10).
    """
    return ceil(
        (2 / eps) * np.log(B / (1 - (1 - gamma * alpha)**(1 / m)))
    )


def select_gamma(M: np.ndarray, alpha: float, eps: float, B: int) -> dict:
    """
    Select gamma by minimizing M_{ell_gamma + ell_cor, k_gamma}
    over a grid of candidate values, following Section 4 of the paper.

    Only configurations satisfying 0 < q < 1 are considered.
    """

    m, n = M.shape
    best = None

    # Search over a grid of candidate gamma values in (0, 1).
    for gamma in np.linspace(0.01, 0.99, 99):
        adjusted_level = (1 - alpha) / (1 - gamma * alpha)

        # Find the quantile orders ell and k corresponding to the smallest
        # entry in M that achieves at least the adjusted coverage level.
        # Skip the current gamma if no such quantile orders exist.
        try:
            ell, k = find_l_star_k_star(M, adjusted_level)
        except ValueError:
            continue

        # Compute the privacy correction and add it to the local quantile order.
        ell_cor = correction(alpha, gamma, eps, B, m)
        order = ell + ell_cor

        # Since q = max(order / n, 0.5), order >= n implies q >= 1.
        # Skip such configurations, as Algorithm 2 requires 0 < q < 1.
        if order >= n:
            continue

        # Evaluate the coverage objective M_{ell_gamma + ell_cor, k_gamma}
        # and retain the configuration if it achieves the smallest value so far.
        coverage = float(M[k - 1, order - 1])

        if best is None or coverage < best['coverage']:
            best = {
                'gamma': float(gamma),
                'ell': ell,
                'k': k,
                'ell_cor': ell_cor,
                'q': max(order / n, 0.5),
                'coverage': coverage
            }

    # Raise an error if none of the candidate gamma values are feasible.
    if best is None:
        raise ValueError(
            f'No feasible gamma found on the grid for epsilon={eps}'
        )

    # Return the feasible configuration with the smallest coverage objective.
    return best


def fedcp2_qq(local_scores: np.ndarray, eps: float, bin_edges: np.ndarray,
              config: dict, rng: np.random.Generator) -> float:
    """
    Return the global differentially private quantile-of-quantiles.
    """

    # Convert local scores to a floating-point NumPy array.
    local_scores = np.asarray(local_scores, dtype=float)
    m = local_scores.shape[0]

    # Each client computes its differentially private quantile
    # using Algorithm 2 with the selected quantile level q.
    client_quantiles = np.array([
        dp_quantile(local_scores[j], config['q'], eps, bin_edges, rng)
        for j in range(m)
    ])

    # The server computes the k_gamma-th smallest local quantile.
    server_quantile = np.partition(
        client_quantiles, config['k'] - 1
    )[config['k'] - 1]

    return float(server_quantile)