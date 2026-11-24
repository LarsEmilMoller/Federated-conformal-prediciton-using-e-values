#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:56:45 2026

@author: livboethrysoe
"""

"""
Implementation of the FCeP-DP algorithm for locally differentially private
federated conformal prediction using e-values.

Includes the truncated Laplace mechanism, client-side privatization,
server-side aggregation and computation of privatized federated e-values.
"""

import numpy as np

# -------------------------------------------------------------------------
# 1. Truncated Laplace mechanism
# -------------------------------------------------------------------------
def truncation_bound(S_max, eps, delta):
    """
    Compute the truncation bound T for the truncated Laplace mechanism.
    """
    return (S_max / eps) * np.log(
        1 + (np.exp(eps) - 1) / (2 * delta)
    )


def truncated_laplace(scale, T, rng):
    """
    Draw a single sample from a centered Laplace distribution
    truncated to [-T, T] using rejection sampling.

    This function is called once per client to generate local privacy noise.
    """
    while True:
        # Draw noise from the ordinary Laplace distribution.
        Z = rng.laplace(0, scale)

        # Return the sample if it lies within [-T, T].
        # Otherwise, discard it and draw a new sample.
        if abs(Z) <= T:
            return float(Z)


# -------------------------------------------------------------------------
# 2. Client-side computations
# -------------------------------------------------------------------------
def client_message(local_scores, S_max, eps, T, rng):
    """
    Compute the privatized local score sum and number of calibration points
    of a client.
    """
    local_scores = np.asarray(local_scores, dtype=float)
    n_k = len(local_scores)

    # Check that the calibration scores satisfy the assumed score bound.
    if not np.all((local_scores >= 0) & (local_scores <= S_max)):
        raise ValueError('Calibration scores must belong to [0, S_max]')

    S_k = np.sum(local_scores)
    Z_k = truncated_laplace(S_max / eps, T, rng)
    S_hat_k = S_k + Z_k

    return float(S_hat_k), n_k


# -------------------------------------------------------------------------
# 3. Server-side aggregation
# -------------------------------------------------------------------------
def aggregate_messages(messages):
    """
    Aggregate the received client messages (S_hat_k, n_k) to obtain the total 
    calibration size N and the aggregated privatized score sum S_hat_G.
    """
    N = sum(n_k for S_hat_k, n_k in messages)
    S_hat_G = sum(S_hat_k for S_hat_k, n_k in messages)

    return N, S_hat_G


# -------------------------------------------------------------------------
# 4. FCeP-DP calibration
# -------------------------------------------------------------------------
def fcep_dp_calibration(local_scores, S_max, eps, delta, rng):
    """
    Execute the client-side computations and server-side aggregation
    of FCeP-DP.
    """
    if S_max <= 0 or eps <= 0 or not 0 < delta < 0.5:
        raise ValueError('Require S_max > 0, eps > 0 and 0 < delta < 0.5')

    K = len(local_scores)
    T = truncation_bound(S_max, eps, delta)

    # Each client computes and sends its privatized score sum and
    # calibration size to the server.
    messages = [
        client_message(local_scores[k], S_max, eps, T, rng)
        for k in range(K)
    ]

    # Aggregate the messages at the server.
    N, S_hat_G = aggregate_messages(messages)

    # Store the quantities needed to compute the privatized e-values.
    return {
        'K': K,
        'N': N,
        'S_hat_G': S_hat_G,
        'S_max': S_max,
        'eps': eps,
        'T': T
    }


# -------------------------------------------------------------------------
# 5. Privatized federated e-values
# -------------------------------------------------------------------------
def fcep_dp_e_values(test_scores, calibration):
    """
    Compute the privatized federated e-values for the given test scores.
    """
    test_scores = np.asarray(test_scores, dtype=float)

    # Extract the quantities obtained during calibration.
    K = calibration['K']
    N = calibration['N']
    S_hat_G = calibration['S_hat_G']
    S_max = calibration['S_max']
    eps = calibration['eps']
    T = calibration['T']

    # Check that the test scores satisfy the assumed score bound.
    if not np.all((test_scores >= 0) & (test_scores <= S_max)):
        raise ValueError('Test scores must belong to [0, S_max]')

    # Compute A(y) for each test score.
    A = S_hat_G + test_scores + (K - 1) * S_max

    # Exclude undefined expressions due to division by zero.
    if np.any(A == 0) or np.any(A - K * T == 0):
        raise ValueError('Undefined denominator in FCeP-DP')

    # Compute the privatized federated e-values.
    e_values = (
        (N + K) * test_scores / A
        * (1 + (4 * K * S_max**2) / (eps**2 * (A - K * T)**2))**(-1)
    )

    return e_values


# -------------------------------------------------------------------------
# 6. Construct prediction set
# -------------------------------------------------------------------------
def acceptance(e_values, alpha):
    """
    Return a Boolean array indicating whether each e-value satisfies
    the FCeP-DP acceptance criterion.
    """
    return e_values < 1 / alpha


def prediction_set(candidate_labels, test_scores, calibration, alpha):
    """
    Construct the prediction set by computing the privatized federated
    e-value for each candidate label and retaining only those that
    satisfy the acceptance criterion.
    """
    e_values = fcep_dp_e_values(test_scores, calibration)

    return np.asarray(candidate_labels)[acceptance(e_values, alpha)]