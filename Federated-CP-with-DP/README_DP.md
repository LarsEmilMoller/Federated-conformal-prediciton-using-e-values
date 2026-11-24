# Federated Conformal Prediction with Differential Privacy

Empirical evaluation and comparison of FedCP²-QQ and FCeP-DP on the Bike and Bio datasets.

## Scripts

| Script | Description |
| --- | --- |
| `prepare_scores_DP.py` | Prepares dataset splits, fits ridge regression models and computes calibration and test scores. |
| `FedCP2_QQ_DP.py` | Implements FedCP-QQ and FedCP²-QQ. |
| `experiment_FedCP2_QQ_DP.py` | Evaluates FedCP²-QQ and the non-private FedCP-QQ baseline using coverage boxplots. |
| `FCeP_DP.py` | Implements FCeP-DP. |
| `experiment_FCeP_DP.py` | Evaluates FCeP-DP using a delta sweep and coverage boxplots. |
| `experiment_compare_DP.py` | Compares FedCP²-QQ and FCeP-DP using an epsilon sweep and a client-count sweep. |

## Running the experiments

Run `prepare_scores_DP.py` first, followed by the desired `experiment_*.py` script. Experimental parameters are specified in Section 1 of each experiment script.

The `cached_data_DP/` directory stores prepared scores, score bounds and coverage matrices for reuse across experiments.
