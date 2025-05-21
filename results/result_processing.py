import os
from glob import glob
from math import inf
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_auc_score

from config import STORAGE_DIR, MY_STORAGE_DIR


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
        'reduce-by-x', 'norm1', 'norm2', 'norm3', 'norm4', 'inf'] = None
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
    if not reduction:
        return traces

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


def get_lira_scores(exp_id: str, target_id: str = 'target', return_full_df=True):

    df = pd.read_csv(os.path.join(MY_STORAGE_DIR, 'lira_scores', exp_id + '_' + target_id))
    if return_full_df:
        return df
    return df['lira_score'].sort_index()


def get_attackr_scores(exp_id: str, target_id: str = 'target', return_full_df=False):
    path = [f for f in glob(f'{MY_STORAGE_DIR}/attackr_perc_scores/{exp_id}_{target_id}')][0]

    try:
        data = pd.read_csv(path)

        attack_score = list(filter(lambda x: x.startswith('attackr'), data.columns))

        if return_full_df:
            return data
        else:
            return 1-data[attack_score].sort_index()
    except Exception as e:
        print(e)

def plot_attackr_roc(exp_id: str, target_id: str = 'target', alphas=np.logspace(-5, 0, 100)):

    try:
        alphas = sorted(alphas)

        tpr_values = []
        fpr_values = []

        for alpha in alphas:
            path = [f for f in glob(f'{MY_STORAGE_DIR}/attackr_{alpha}_*/{exp_id}_{target_id}')][0]
            data = pd.read_csv(path)

            y_eval = data['target_trained_on'].astype(int)
            y_pred = data['preds']
            tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            tpr_values.append(tpr)
            fpr_values.append(fpr)

            roc_auc = roc_auc_score(y_eval, y_pred)
            print(roc_auc)

        tpr_values.insert(0, 0)
        fpr_values.insert(0, 0)
        tpr_values.append(1)
        fpr_values.append(1)

        auc_value = round(auc(x=fpr_values, y=tpr_values), 5)

        fig, ax = plt.subplots()
        ax.plot(fpr_values,
                tpr_values,
                linewidth=2.0,
                color='b',
                label=f'AUC = {auc_value}')
        plt.xscale('log')
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_ylim([0.0, 1.1])
        ax.legend(loc='lower right')
    except Exception as e:
         print(e)

def get_rmia_scores(exp_id: str, target_id: str = 'target', return_full_df=False):
    path = [f for f in glob(f'{MY_STORAGE_DIR}/rmia_2.0_scores/{exp_id}_{target_id}')][0]

    try:
        df = pd.read_csv(path)
        if return_full_df:
            return df
        else:
            attack_score = list(filter(lambda x: 'rmia' in x, df.columns))
            return df[attack_score].sort_index()
    except Exception as e:
         print(e)

def print_overall_tpr_at_fpr(df: pd.DataFrame, target_col="lira_score"):
    fpr, tpr, _thresholds = metrics.roc_curve(df['target_trained_on'], df[target_col], drop_intermediate=False)
    plt.plot(fpr, tpr)
    plt.xlim(left=0)
    plt.xscale('log')

    levels = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    print("AUC: ",metrics.roc_auc_score(df['target_trained_on'], df[target_col]))
    for level in levels:
        low = tpr[np.where(fpr<=level)[0][-1]]
        print("TPR@FPR="+str(level), low)


def get_overall_tpr_at_fpr(df: pd.DataFrame, query_fpr: float, target_col="lira_score"):
    fpr, tpr, _thresholds = metrics.roc_curve(df['target_trained_on'], df[target_col], drop_intermediate=False)
    return tpr[np.where(fpr<=query_fpr)[0][-1]]


# Assign each sample to equally sized bins based on similar `binning_col` values
def create_bins(df: pd.DataFrame, bins: int = 100, bin_separately: bool = True, binning_col: str = 'avg_norm') -> pd.Series:

    if bin_separately:
        members = df[df['target_trained_on'] == True].copy()
        members['bin'] = pd.qcut(members[binning_col].rank(method='first'), bins, labels=False)

        non_members = df[df['target_trained_on'] == False].copy()
        non_members['bin'] = pd.qcut(non_members[binning_col].rank(method='first'), bins, labels=False)

        result = pd.concat([members, non_members]).sort_index()['bin']
    else:
        result = pd.qcut(df[binning_col], bins, labels=False)

    return result
