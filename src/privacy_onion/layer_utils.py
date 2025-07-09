import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
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
    get_attackr_scores,
    get_lira_scores,
    get_rmia_scores,
    get_trace_reduction,
)
from loss_traces.results.utils import (
    make_precision_recall_at_k_df_single_threshold,
    make_precision_recall_at_k_df,
)
from loss_traces.attacks import AttackConfig, RMIAAttack
from sklearn.metrics import roc_curve


def get_lt_iqr(exp_id):
    saves = torch.load(f"{MODEL_DIR}/{exp_id}/target", weights_only=False)

    df = pd.DataFrame(get_trace_reduction(exp_id, reduction="iqr"), columns=["lt_iqr"])
    df['og_idx'] = df.index
    df['target_trained_on'] = df.index.isin(saves['trained_on_indices'])
    
    return df 

def plot_log_lt_iqr(exp_id, layer, top_k=0.05, save_fig=True):
    plt.style.use("plot_style.mplstyle")
    os.makedirs(f"{STORAGE_DIR}/figures", exist_ok=True)

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
        plt.savefig(f"figures/lt_iqr_histogram_layer_{layer}.png")

    plt.show()

def plot_lt_iqr_no_log(exp_id, layer, top_k=0.05, save_fig=True):
    plt.style.use("plot_style.mplstyle")

    os.makedirs(f"{STORAGE_DIR}/figures", exist_ok=True)

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
        plt.savefig(f"figures/lt_iqr_histogram_layer_{layer}_no_log.png")

    plt.show()


def plot_kde(exp_ids, top_k=0.05, save_fig=True):
    plt.style.use("plot_style.mplstyle")
    os.makedirs(f"{STORAGE_DIR}/figures", exist_ok=True)

    
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

    if save_fig:
        fig.savefig(f"{STORAGE_DIR}/figures/lt_iqr_kde_overlay.png")

    plt.show()


def save_layer_target_indices(exp_id, layer, exp_path, top_k=0.05):
    os.makedirs(f"{STORAGE_DIR}/layer_target_indices", exist_ok=True)

    save_path = f"{STORAGE_DIR}/layer_target_indices/{exp_path}"
    os.makedirs(save_path, exist_ok=True)

    df = get_lt_iqr(exp_id)

    members = df[df['target_trained_on'] == True]
    top_score = members.sort_values(by='lt_iqr', ascending=False)

    idx = int(len(top_score) * top_k)

    vulnerable = top_score[:idx]['og_idx']
    safe = top_score[idx:]['og_idx']

    # create safe fulset that is all prev_ safe fulset - vulnerable if layer > 0, if not then len(idx) - vulnerable
    if layer > 0:
        prev_safe = pd.read_csv(f"{save_path}/layer_{layer - 1}_full_safe.csv")
        full_safe = prev_safe[~prev_safe['og_idx'].isin(vulnerable)]

    else:
        full_safe = pd.DataFrame({'og_idx': np.arange(len(df))})
        full_safe = full_safe[~full_safe['og_idx'].isin(vulnerable)]
    
    # vulnerable.to_csv(
    #     f"{save_path}/layer_{layer}_vulnerable.csv",
    #     index=False)
    full_safe.to_csv(
        f"{save_path}/layer_{layer}_full_safe.csv",
        index=False)
    
    safe.to_csv(f"{save_path}/layer_{layer}_safe.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process layer target indices.")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--top_k", type=float, default=0.05, help="Top k percentage for target indices")
    parser.add_argument("--exp_path", type=str, help="Path to the experiment directory")

    args = parser.parse_args()

    save_layer_target_indices(args.exp_id, args.layer, args.exp_path, args.top_k)