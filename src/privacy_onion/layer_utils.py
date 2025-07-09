import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import argparse

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
    plt.title(f"LT-IQR Layer {layer} ({len(lt_iqr)} samples)")
    plt.xlabel('LT-IQR Score (log scale)')
    plt.ylabel('Frequency')
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"figures/lt_iqr_histogram_layer_{layer}.png")

    plt.show()


def plot_log_lira(exp_id, layer, top_k=0.05, save_fig=True):
    df = get_lira_scores(exp_id, n_shadows=32)

    #plot lira
    lira_scores = df["lira_score"].values
    lira_scores = lira_scores[lira_scores > 0]
    log_bins = np.logspace(np.log10(lira_scores.min()), np.log10(lira_scores.max()), 50)
    plt.figure(figsize=(10, 6))
    plt.hist(lira_scores, bins=log_bins, alpha=0.7, edgecolor='black')
    plt.xscale('log')
    plt.title(f"LIRA Histogram — Layer 0 ({len(lira_scores)} samples)")
    plt.xlabel('LIRA Score (log scale)')
    plt.ylabel('Frequency')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"figures/lira_histogram_layer_{layer}.png")
    plt.show()


def save_layer_target_indices(exp_id, layer, top_k=0.05):
    os.makedirs(f"{STORAGE_DIR}/layer_target_indices", exist_ok=True)

    save_path = f"{STORAGE_DIR}/layer_target_indices/{exp_id}"
    os.makedirs(save_path, exist_ok=True)

    df = get_lt_iqr(exp_id)

    members = df[df['target_trained_on'] == True]
    top_score = members.sort_values(by='lt_iqr', ascending=False)

    idx = int(len(top_score) * top_k)

    vulnerable = top_score[:idx]['og_idx']
    safe = top_score[idx:]['og_idx']

    vulnerable.to_csv(
        f"{save_path}/layer_{layer}_vulnerable.csv",
        index=False)
    
    safe.to_csv(f"{save_path}/layer_{layer}_safe.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process layer target indices.")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--top_k", type=float, default=0.05, help="Top k percentage for target indices")

    args = parser.parse_args()

    save_layer_target_indices(args.exp_id, args.layer, args.top_k)