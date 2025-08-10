import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import pickle
import argparse

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from loss_traces.config import STORAGE_DIR, MODEL_DIR

# from loss_traces.config import MODEL_DIR
from loss_traces.data_processing.data_processing import (
    get_no_shuffle_train_loader,
    get_num_classes,
)
from loss_traces.models.model import load_model
from loss_traces.results.final_model_metrics import get_final_model_metrics
from loss_traces.results.result_processing import (
    get_trace_reduction,
)
from loss_traces.results.utils import (
    make_precision_recall_at_k_df_single_threshold,
    make_precision_recall_at_k_df,
)
from loss_traces.results.result_processing import get_lira_scores
from loss_traces.attacks import AttackConfig, RMIAAttack
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics


def get_lt_iqr(exp_id):
    saves = torch.load(f"{MODEL_DIR}/{exp_id}/target", weights_only=False)

    df = pd.DataFrame(get_trace_reduction(exp_id, reduction="iqr"), columns=["lt_iqr"])
    df['og_idx'] = df.index
    df['target_trained_on'] = df.index.isin(saves['trained_on_indices'])
    
    return df 

def plot_log_lt_iqr(exp_id, layer, top_k=0.05, save_fig=True):
    plt.style.use("plot_style.mplstyle")
    os.makedirs(f"./figures", exist_ok=True)

    df = get_lt_iqr(exp_id)

    members = df[df['target_trained_on'] == True]
    top_score = members.sort_values(by='lt_iqr', ascending=False)

    idx = int(len(top_score) * top_k)

    lt_iqr = top_score["lt_iqr"].values
    threshold = lt_iqr[idx]

    log_bins = np.logspace(np.log10(lt_iqr.min()), np.log10(lt_iqr.max()), 50)
    plt.figure(figsize=(10, 6))
    plt.hist(lt_iqr, bins=log_bins, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1, label=f'Threshold')
    plt.legend()
    plt.xscale('log')
    plt.xlabel('LT-IQR Score (log scale)')
    plt.ylabel('Frequency')
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"./figures/lt_iqr_histogram_layer_{layer}.png")

    plt.show()

def plot_lt_iqr_no_log(exp_id, layer, top_k=0.05, save_fig=True):
    plt.style.use("plot_style.mplstyle")

    os.makedirs(f"./figures", exist_ok=True)

    df = get_lt_iqr(exp_id)

    members = df[df['target_trained_on'] == True]
    top_score = members.sort_values(by='lt_iqr', ascending=False)

    idx = int(len(top_score) * top_k)

    lt_iqr = top_score["lt_iqr"].values
    threshold = lt_iqr[idx]

    plt.figure(figsize=(10, 6))

    plt.xlim(lt_iqr.min(), lt_iqr.max())
    plt.hist(lt_iqr, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1, label=f'Threshold')
    plt.legend()
    plt.xlabel('LT-IQR Score')
    plt.ylabel('Frequency')
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"./figures/lt_iqr_histogram_layer_{layer}_no_log.png")

    plt.show()


def plot_kde(exp_ids, top_k=0.05, save_name=None):
    plt.style.use("plot_style.mplstyle")
    os.makedirs(f"./figures", exist_ok=True)
    
    num_layers = len(exp_ids)
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=num_layers - 1)
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer, exp_id in enumerate(exp_ids):
        df = get_lt_iqr(exp_id)
        members = df[df['target_trained_on'] == True]
        top_score = members.sort_values(by='lt_iqr', ascending=False)
        idx = int(len(top_score) * top_k)
        lt_iqr = top_score["lt_iqr"].values[:idx]

        color = cmap(norm(layer))
        sns.kdeplot(lt_iqr, ax=ax, color=color, fill=False, linewidth=2)

    # Axis labels
    ax.set_xlabel('LT-IQR Score')
    ax.set_ylabel('Density')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for older matplotlib versions
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Layer")

    fig.tight_layout()

    if save_name:
        fig.savefig(f"./figures/{save_name}.png")

    plt.show()

def plot_kde_custom(exp_ids, top_k=0.05, labels=None, label_name="Layer", save_name=None):
    plt.style.use("plot_style.mplstyle")
    os.makedirs(f"./figures", exist_ok=True)

    if labels is not None:
        # Use the range of labels for color normalization
        vmin = min(labels)
        vmax = max(labels)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        num_layers = len(labels)
    else:
        num_layers = len(exp_ids)
        norm = mcolors.Normalize(vmin=0, vmax=num_layers - 1)

    cmap = cm.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, exp_id in enumerate(exp_ids):
        df = get_lt_iqr(exp_id)
        members = df[df['target_trained_on'] == True]
        top_score = members.sort_values(by='lt_iqr', ascending=False)
        idx = int(len(top_score) * top_k)
        lt_iqr = top_score["lt_iqr"].values[:idx]

        if labels is not None:
            color = cmap(norm(labels[i]))
        else:
            color = cmap(norm(i))
        sns.kdeplot(lt_iqr, ax=ax, color=color, fill=False, linewidth=2)

    ax.set_xlabel('LT-IQR Score')
    ax.set_ylabel('Density')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(label_name)

    fig.tight_layout()

    if save_name:
        fig.savefig(f"./figures/{save_name}.png")

    plt.show()

def plot_kde_lira(exp_ids, save_name=None):
    plt.style.use("plot_style.mplstyle")
    os.makedirs(f"{STORAGE_DIR}/figures", exist_ok=True)

    num_layers = len(exp_ids)
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=num_layers - 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer, exp_id in enumerate(exp_ids):
        df = get_lira_scores(exp_id, n_shadows=32)

        lira_score = df["lira_score"].values
        lira_score = np.log10(lira_score + 1e-10)  # Apply log transformation

        color = cmap(norm(layer))
        sns.kdeplot(lira_score, ax=ax, color=color, fill=False, linewidth=2)

    ax.set_xlabel('LiRA Score')
    ax.set_ylabel('Density')
    plt.xlim(left=-5, right=5)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for older matplotlib versions
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Layer")

    fig.tight_layout()

    if save_name:
        fig.savefig(f"./figures/{save_name}.png")

    plt.show()


def save_layer_target_indices(exp_id, layer, exp_path, top_k=0.05, random=False):
    os.makedirs(f"{STORAGE_DIR}/layer_target_indices", exist_ok=True)

    save_path = f"{STORAGE_DIR}/layer_target_indices/{exp_path}"
    os.makedirs(save_path, exist_ok=True)

    df = get_lt_iqr(exp_id)

    members = df[df['target_trained_on'] == True]
    top_score = members.sort_values(by='lt_iqr', ascending=False)

    idx = int(len(top_score) * top_k)
    if random is False:
        vulnerable = top_score[:idx]['og_idx']
        safe = top_score[idx:]['og_idx']
    else:
        np.random.seed(42)  # For reproducibility
        og_idx = top_score['og_idx'].values
        random_sample = np.random.choice(og_idx, size=idx, replace=False)
        vulnerable = top_score[top_score['og_idx'].isin(random_sample)]['og_idx']
        safe = top_score[~top_score['og_idx'].isin(random_sample)]['og_idx']


    # create safe fulset that is all prev_ safe fulset - vulnerable if layer > 0, if not then len(idx) - vulnerable

    if layer > 0:
        prev_safe = pd.read_pickle(f"{save_path}/layer_{layer - 1}_full_safe.pkl")
        full_safe = prev_safe[~prev_safe['og_idx'].isin(vulnerable)]
    else:
        full_safe = pd.DataFrame({'og_idx': np.arange(len(df))})
        full_safe = full_safe[~full_safe['og_idx'].isin(vulnerable)]

    full_safe.to_pickle(
        f"{save_path}/layer_{layer}_full_safe.pkl")
    
    safe.to_pickle(f"{save_path}/layer_{layer}_safe.pkl")

def tpr_at_fpr(exp_id, fpr=0.001, target_id='target'):
    df = get_lira_scores(exp_id, n_shadows=32, target_id=target_id)

    labels = df["target_trained_on"]
    results = []
    scores = df['lira_score']
    
    fpr_values, tpr_values, thresholds = metrics.roc_curve(labels, scores)
    idx = (np.abs(fpr_values - fpr)).argmin()
    tpr_at_fpr = tpr_values[idx]
    threshold_at_fpr = thresholds[idx]

    y_pred = scores > thresholds[idx]
    y_true = labels.astype(bool)
    true_positives = np.sum((y_pred == 1) & (y_true == 1))

    precision = metrics.precision_score(labels, scores > thresholds[idx])
    recall = metrics.recall_score(labels, scores > thresholds[idx])
    auc = metrics.roc_auc_score(labels, scores)
    
    results.append({
        'attack': 'LiRA',
        f'tpr_at_fpr_{fpr}': tpr_at_fpr,
        'tps': true_positives,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'threshold_at_fpr': threshold_at_fpr
    })

    result_df = pd.DataFrame(results)
    return result_df


def tpr_at_threshold(df, threshold=0.5):

    labels = df["target_trained_on"]
    results = []
    scores = df['lira_score']
    
    y_pred = scores > threshold
    y_true = labels.astype(bool)
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    
    precision = metrics.precision_score(labels, y_pred)
    recall = metrics.recall_score(labels, y_pred)
    auc = metrics.roc_auc_score(labels, scores)
    
    tpr_at_threshold = np.sum(y_true & y_pred) / np.sum(y_true)

    results.append({
        'attack': 'LiRA',
        'tpr_at_threshold': tpr_at_threshold,
        'tps': true_positives,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'threshold': threshold
    })

    result_df = pd.DataFrame(results)
    return result_df

def calculate_roc(df: pd.DataFrame, cols=["lira_score", "attackr_score", "rmia_score"]):
    """
    Calculates FPR and TPR for each column in cols and adds them as new columns to the DataFrame.
    Returns a dictionary with attack names as keys and (fpr, tpr) tuples as values.
    """
    roc_data = {}
    for target_col in cols:
        fpr, tpr, _ = metrics.roc_curve(df['target_trained_on'], df[target_col])
        auc = metrics.auc(fpr, tpr)
        roc_data[target_col] = (fpr, tpr, auc)
    return roc_data


def plot_roc(exp_id, save_path=None):
    """
    Plots ROC curves from roc_data (dict of {label: (fpr, tpr)}).
    Optionally, extra_curves (list of (fpr, tpr)) and extra_labels (list of str)
    can be provided.
    """
    df = get_lira_scores(exp_id, n_shadows=32)
    roc_data = calculate_roc(df, cols=["lira_score"])
    plt.figure(figsize=(8, 6))
    for label, (fpr, tpr, auc) in roc_data.items():
        if label == "rmia_score":
            label = "RMIA"
        elif label == "attackr_score":
            label = "AttackR"
        elif label == "lira_score":
            label = "LiRA"
        plt.plot(fpr, tpr, label=label)
    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        color='gray',
        label='Random Guessing'
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(bottom=0.001)
    plt.xlim(left=0.001)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()    


def plot_multi_roc(exp_ids_target, save_name=None):
    """
    Plots ROC curves from roc_data (dict of {label: (fpr, tpr)}).
    Optionally, extra_curves (list of (fpr, tpr)) and extra_labels (list of str)
    can be provided.
    """

    plt.figure(figsize=(8, 6))
    plt.style.use("plot_style.mplstyle")
    for exp_id, target, label in exp_ids_target:
        df = get_lira_scores(exp_id, target_id=target, n_shadows=32)
        roc_data = calculate_roc(df, cols=["lira_score"])
        for _, (fpr, tpr, auc) in roc_data.items():
            if label == "Ideal":
                plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})", linestyle='--')
            else:
                plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})")


    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(bottom=0.001)
    plt.xlim(left=0.001)
    if save_name:
        plt.savefig(f"./figures/{save_name}.png", bbox_inches='tight', dpi=300)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process layer target indices.")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--top_k", type=float, default=0.05, help="Top k percentage for target indices")
    parser.add_argument("--exp_path", type=str, help="Path to the experiment directory")
    parser.add_argument(
        "--random",
        action="store_true",
        default=False,
        help="If set, randomly select target indices instead of using top k",
    )

    args = parser.parse_args()

    save_layer_target_indices(args.exp_id, args.layer, args.exp_path, args.top_k, args.random)