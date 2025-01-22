import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve
from tqdm import tqdm


def make_precision_recall_at_k_df(
        scores_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        k_frac: float = 0.01,
        membership_col: str = "target_trained_on"
):
    y_true = ground_truth_df[membership_col].astype(int)
    y_score = ground_truth_df["lira_score"]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    total_members = len(scores_df[scores_df[membership_col]])
    k = int(k_frac * total_members)
    res = defaultdict(list)
    predicted = {}
    for col in scores_df.columns:
        if col == membership_col:
            continue

        predicted[col + "_desc"] = set(scores_df[scores_df[membership_col]]
                                       [col].sort_values(ascending=False).head(k).index)
        predicted[col + "_asc"] = set(scores_df[scores_df[membership_col]]
                                      [col].sort_values(ascending=True).head(k).index)

    for xfpr, xtpr, xt in tqdm(zip(fpr, tpr, thresholds)):
        ground_truth_positives = ground_truth_df[(ground_truth_df['lira_score'] > xt) & (
            ground_truth_df[membership_col] == True)]
        positive_indices = set(ground_truth_positives.index)

        res["fpr"].append(xfpr)
        res["tpr"].append(xtpr)
        res["positives"].append(len(positive_indices))
        res["precision_random_guess"].append(len(positive_indices) / total_members)
        res["recall_random_guess"].append(k / total_members)

        for col, topk_predicted in predicted.items():
            overlap = len(positive_indices & topk_predicted)
            res["overlap_" + col].append(overlap)
            res["precision_" + col].append(overlap / k)
            
            if len(positive_indices) > 0:
                res["recall_" + col].append(overlap / len(positive_indices))
            else:
                res["recall_" + col].append(None)

    return pd.DataFrame(res)


def make_precision_recall_at_k_with_union(
        scores_df: pd.DataFrame,
        ground_truth_dfs: list[pd.DataFrame],
        fpr_threshold: float,
        k_frac: float = 0.01,
        membership_col: str = "target_trained_on"
):  
    if not isinstance(ground_truth_dfs, list):
        raise ValueError()
    
    all_positive_indices = set()

    for ground_truth_df in ground_truth_dfs:
        y_true = ground_truth_df[membership_col].astype(int)
        y_score = ground_truth_df["score"]
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        idx = np.where(fpr > fpr_threshold)[0][0]
        ground_truth_positives = ground_truth_df[(ground_truth_df['score'] > thresholds[idx]) & (
            ground_truth_df[membership_col] == True)]
        positive_indices = set(ground_truth_positives.index)
        all_positive_indices |= positive_indices

    total_members = len(scores_df[scores_df[membership_col]])
    k = int(k_frac * total_members)
    res = {}
    predicted = {}
    for col in scores_df.columns:
        if col == membership_col:
            continue

        predicted[col + "_desc"] = set(scores_df[scores_df[membership_col]]
                                       [col].sort_values(ascending=False).head(k).index)
        predicted[col + "_asc"] = set(scores_df[scores_df[membership_col]]
                                      [col].sort_values(ascending=True).head(k).index)

    res["positives"] = len(all_positive_indices)
    res["precision_random_guess"] = len(all_positive_indices) / total_members
    res["recall_random_guess"] = k / total_members

    for col, topk_predicted in predicted.items():
        overlap = len(all_positive_indices & topk_predicted)
        res["overlap_" + col] = overlap
        res["precision_" + col] = overlap / k
        
        if len(all_positive_indices) > 0:
            res["recall_" + col] = overlap / len(all_positive_indices)
        else:
            res["recall_" + col] = None

    return res


def get_precision_at_fpr(precision_df: pd.DataFrame, fpr: float):
    return dict(precision_df[precision_df["fpr"] > fpr].sort_values(by='fpr', ascending=True).iloc[0])
