import os
from glob import glob
from math import inf
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import stats
from sklearn import metrics

from config import MODEL_DIR, STORAGE_DIR, MY_STORAGE_DIR



def _is_default_index(df: pd.DataFrame) -> bool:
    ## Check if the first column appears to be a default sequential index.
    return (df.iloc[0, 0] == 0 and
            df.iloc[1, 0] == 1 and
            df.iloc[2, 0] == 2)


def _select_columns(df: pd.DataFrame, first: Optional[int] = None, last: Optional[int] = None) -> pd.DataFrame:
    select = list(range(len(df.columns)))

    if last is not None:
        select = select[:last]
    if first is not None:
        select = select[first:]

    return df.iloc[:, select]

def get_reduced_data(
        path: str,
        first: Optional[int] = None,
        last: Optional[int] = None,
        reduction: Literal['mean', 'slope', 'iqr', 'mid-end', 'delta/mid',
        'reduce-by-x', 'norm1', 'norm2', 'norm3', 'norm4', 'inf'] = 'mean'
) -> pd.Series:
    """
    Read and process data from a Parquet file with various reduction methods.

    Parameters:
    -----------
    path : str
        Path to the Parquet file
    first : int, optional
        Starting column index to select
    last : int, optional
        Ending column index to select
    reduction : str, optional
        Method of reducing the data. Defaults to 'mean'.
        Supported methods:
        - 'mean': Average across selected columns
        - 'slope': Linear regression slope
        - 'iqr': Interquartile range
        - 'mid-end': Difference between mid and end values
        - 'delta/mid': Normalized change from mid to end
        - 'reduce-by-x': Time to reach threshold
        - 'norm{1-4}': Lp norms
        - 'inf': Infinity norm

    Returns:
    --------
    pd.Series
        Reduced data series
    """
    # Read the Parquet file
    df = pd.read_parquet(path)

    # Drop extra index column if it looks like a default index
    if _is_default_index(df):
        df = df.drop(df.columns[0], axis=1)

    # Select columns based on first and last parameters
    traces = _select_columns(df, first, last)

    # Apply the specified reduction method
    return _apply_reduction(traces, reduction)

def _apply_reduction(traces: pd.DataFrame, reduction: str) -> pd.Series:
    """
    Apply the specified reduction method to the traces.

    Parameters:
    -----------
    traces : pd.DataFrame
        Input DataFrame of traces
    reduction : str
        Reduction method to apply

    Returns:
    --------
    pd.Series
        Reduced data
    """
    reduction_methods = {
        'mean': _reduction_mean,
        'slope': _reduction_slope,
        'iqr': _reduction_iqr,
        'mid-end': _reduction_mid_end,
        'delta/mid': _reduction_delta_mid,
        'reduce-by-x': _reduction_reduce_by_x,
        'norm1': lambda x: np.linalg.norm(x, ord=1, axis=1),
        'norm2': lambda x: np.linalg.norm(x, ord=2, axis=1),
        'norm3': lambda x: np.linalg.norm(x, ord=3, axis=1),
        'norm4': lambda x: np.linalg.norm(x, ord=4, axis=1),
        'inf': lambda x: np.linalg.norm(x, ord=inf, axis=1)
    }

    if reduction not in reduction_methods:
        raise ValueError(f"Unsupported reduction method: {reduction}")

    return reduction_methods[reduction](traces)


def _reduction_mean(traces: pd.DataFrame) -> pd.Series:
    """Calculate mean across columns."""
    return traces.mean(axis=1)


def _reduction_slope(traces: pd.DataFrame) -> pd.Series:
    """Calculate linear regression slope."""
    select = list(range(traces.shape[1]))
    return traces.apply(lambda x: stats.linregress(select, x)[0], axis=1)


def _reduction_iqr(traces: pd.DataFrame) -> pd.Series:
    """Calculate interquartile range."""
    q1 = traces.quantile(0.25, axis=1)
    q3 = traces.quantile(0.75, axis=1)
    return q3 - q1


def _reduction_mid_end(traces: pd.DataFrame) -> pd.Series:
    """Calculate difference between mid and end values."""
    mid = traces.iloc[:, traces.shape[1] // 2]
    end = traces.iloc[:, -1]
    return mid - end


def _reduction_delta_mid(traces: pd.DataFrame) -> pd.Series:
    """Calculate normalized change from mid to end."""
    mid = traces.iloc[:, traces.shape[1] // 2]
    end = traces.iloc[:, -1]
    return 1 - (end / mid)


def _reduction_reduce_by_x(traces: pd.DataFrame) -> pd.Series:
    """Calculate time to reach threshold."""
    thresholds = traces.iloc[:, 0] * 0.000001

    def calc_time_steps(s: pd.Series, threshold: float) -> int:
        """Find index where series drops below threshold."""
        below_threshold = s <= threshold
        return np.argmax(below_threshold) if np.any(below_threshold) else -1

    return traces.apply(lambda row: calc_time_steps(row, thresholds[row.name]), axis=1)


def get_trace_reduction(exp_id: str, target_id: str = None, first: int = None, last: int = None, trace_type="losses", reduction="mean"):
    base_name = exp_id + '_' + target_id if target_id else exp_id
    path = os.path.join(STORAGE_DIR, trace_type, base_name + '.pq')
    return get_reduced_data(path, first, last, reduction=reduction)


def get_lira_scores(exp_id: str, target_id: str = 'target'):
    return pd.read_csv(os.path.join(STORAGE_DIR, 'lira_scores', exp_id + '_' + target_id))


def get_attackr_scores(exp_id: str, target_id: str = 'target'):
     path = [f for f in glob(f'{MY_STORAGE_DIR}/{target_id}/attack_results_*')][0]
     device = "cpu"
     try:
         with np.load(path, allow_pickle=True) as data:
             # normalise train percentiles
             train_percs = data["train_percs"][()]
             test_percs = data["test_percs"][()]

             train_percs = pd.Series({i: n/100 for i,n in train_percs.items()})
             test_percs = pd.Series({i: n/100 for i,n in test_percs.items()})

         dir = os.path.join(MODEL_DIR, exp_id)

         saves = torch.load(os.path.join(dir, target_id), map_location=device)

         train_idx = saves["trained_on_indices"]
         test_idx = list(set(range(50000)) - set(train_idx))

         train_percs.index = train_idx
         test_percs.index = test_idx

         return pd.concat([train_percs, test_percs]).sort_index()
     except Exception as e:
         print(e)


def print_overall_tpr_at_fpr(df: pd.DataFrame):
    fpr, tpr, _thresholds = metrics.roc_curve(df['target_trained_on'], df['lira_score'], drop_intermediate=False)
    levels = [0.1, 0.01, 0.001, 0.0001]
    print("AUC: ",metrics.roc_auc_score(df['target_trained_on'], df['lira_score']))
    for level in levels:
        low = tpr[np.where(fpr<=level)[0][-1]]
        print("TPR@FPR="+str(level), low)

def get_overall_tpr_at_fpr(df: pd.DataFrame, query_fpr: float, target_col="lira_score"):
    fpr, tpr, _thresholds = metrics.roc_curve(df['target_trained_on'], df[target_col], drop_intermediate=False)
    return tpr[np.where(fpr<=query_fpr)[0][-1]]

