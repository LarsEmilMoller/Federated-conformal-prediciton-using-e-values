#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:43:50 2026

@author: livboethrysoe
"""

"""
Prepare and cache all calibration and test absolute residuals for the empirical
evaluation of the differentially private federated conformal prediction
algorithms FedCP2-QQ and FCeP-DP.

The absolute residuals are bounded by S_max, defined as the maximum
absolute residual on the training set for each random split.

Following Humbert et al. (2023), Section 5.2, we use split conformal prediction
with Ridge regression and standardized features. The regularization parameter
is selected using cross-validation. To solve the Ridge regression optimization
problem, we use singular value decomposition (SVD).

This script only needs to be executed once per dataset and training setup.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -------------------------------------------------------------------------
# 1. Settings
# -------------------------------------------------------------------------
# Data and experimental settings
DATASET = 'both'  # 'bike', 'bio', or 'both'
SPLITS = 20
SEED = 42

# Ridge regression settings
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
RIDGE_CV_FOLDS = 5

# Caching settings
CACHE_DIR = Path(__file__).resolve().parent / 'cached_data_DP'
OVERWRITE = False       # Whether to regenerate previously cached scores


# -------------------------------------------------------------------------
# 2. Load and split data
# -------------------------------------------------------------------------
def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load the 'bike' and 'bio' datasets used in Humbert et al. (2023).
    Return the feature matrix X and response vector y.

    Bike: y is the hourly number of bicycle rentals.
    Bio: y is the RMSD between a candidate protein structure and its
         reference structure.
    """

    if name == 'bike':
        url = ('https://raw.githubusercontent.com/yromano/cqr/master/'
               'datasets/bike_train.csv')
        df = pd.read_csv(url)

        # One-hot encode the categorical variables 'season' and 'weather'
        # following Romano et al. (2019).
        season = pd.get_dummies(df['season'], prefix='season', dtype=float)
        weather = pd.get_dummies(df['weather'], prefix='weather', dtype=float)
        df = pd.concat([df, season, weather], axis=1)
        df = df.drop(columns=['season', 'weather'])

        # Extract numerical temporal features from 'datetime'.
        dates = pd.to_datetime(df['datetime'])
        df['hour'] = dates.dt.hour
        df['day'] = dates.dt.dayofweek
        df['month'] = dates.dt.month
        df['year'] = dates.dt.year.map({2011: 0, 2012: 1})

        # Remove the original timestamp and variables that reveal the
        # response variable (count = casual + registered).
        df = df.drop(columns=['datetime', 'casual', 'registered'])

        # Construct X and y.
        X = df.drop(columns=['count']).astype(np.float32)
        y = df['count'].to_numpy(dtype=np.float32)
        return X, y

    elif name == 'bio':
        ds = fetch_openml(data_id=42903, target_column='RMSD', as_frame=True)

        # All explanatory variables are already numerical.
        # Construct X and y.
        X = ds.data.copy()
        y = ds.target.to_numpy(dtype=float)
        return X, y

    else:
        raise ValueError('Dataset must be bike or bio')


def split_data(X, y, split_seed):
    """
    Randomly split the data 40/40/20 between training, calibration and testing.
    """

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.6, random_state=split_seed)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1/3, random_state=split_seed)

    return X_train, y_train, X_cal, y_cal, X_test, y_test


# -------------------------------------------------------------------------
# 3. Train Ridge regression model
# -------------------------------------------------------------------------
def train_ridge_model(X_train, y_train):
    """
    Train a Ridge regression model with standardized explanatory variables.
    Select the regularization parameter using cross-validation.
    """

    # Standardize features and solve the Ridge optimization problem using SVD.
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('ridge', Ridge(solver='svd')),
    ])

    # Select the regularization parameter by minimizing the average
    # validation MSE.
    model = GridSearchCV(
        estimator=pipeline,
        param_grid={'ridge__alpha': RIDGE_ALPHAS},
        cv=RIDGE_CV_FOLDS,
        scoring='neg_mean_squared_error',
        refit=True,
    )

    model.fit(X_train, y_train)
    print(f"Selected Ridge alpha: {model.best_params_['ridge__alpha']:.6g}")

    return model


# -------------------------------------------------------------------------
# 4. Compute and cache conformal scores
# -------------------------------------------------------------------------
def compute_scores(model, X, y):
    """
    Compute the absolute residual scores using the fitted predictor.
    """
    return np.abs(np.asarray(y) - model.predict(X))


def prepare_scores(name):
    """
    Train one predictor per random split and compute and cache all
    corresponding bounded calibration and test absolute residuals
    and the score bound S_max.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f'{name}_scores.npz'

    # Reuse cached scores if the experimental settings are unchanged.
    if path.exists() and not OVERWRITE:
        with np.load(path, allow_pickle=False) as data:
            matches = ('S_max' in data.files and
                       int(data['seed']) == SEED and
                       len(data['cal_scores']) == SPLITS and
                       np.array_equal(data['ridge_grid'], RIDGE_ALPHAS) and
                       int(data['cv_folds']) == RIDGE_CV_FOLDS)

        if not matches:
            raise ValueError(
                f'{path} has different settings; set OVERWRITE=True to regenerate'
            )

        print(f'Using existing {path}')
        return path

    X, y = load_dataset(name)
    cal, test, bounds, selected = [], [], [], []

    # Train the predictor and compute calibration and test scores for each split.
    for split in range(SPLITS):
        X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_data(X, y, SEED + split)
        model = train_ridge_model(X_tr, y_tr)

        # Determine S_max as the maximum training score.
        train_scores = compute_scores(model, X_tr, y_tr)
        S_max = float(np.max(train_scores))

        # Compute calibration and test scores and bound them by S_max.
        cal.append(np.minimum(compute_scores(model, X_cal, y_cal), S_max))
        test.append(np.minimum(compute_scores(model, X_te, y_te), S_max))

        bounds.append(S_max)
        selected.append(model.best_params_['ridge__alpha'])

        print(f'{name}: split {split + 1}/{SPLITS}')

    # Save all computed scores and the corresponding experimental settings.
    np.savez_compressed(
        path,
        cal_scores=np.stack(cal),
        test_scores=np.stack(test),
        S_max=np.asarray(bounds),
        selected_ridge_alphas=selected,
        seed=SEED,
        split_seeds=np.arange(SEED, SEED + SPLITS),
        ridge_grid=RIDGE_ALPHAS,
        cv_folds=RIDGE_CV_FOLDS,
        dataset=name
    )

    print(f'Saved {path}')
    return path


# -------------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------------
def main():
    for name in ('bike', 'bio') if DATASET == 'both' else (DATASET,):
        prepare_scores(name)


if __name__ == '__main__':
    main()
