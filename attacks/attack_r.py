import argparse
import os
from glob import glob
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score, auc, confusion_matrix

from torch.utils.data import Subset, DataLoader

from config import MODEL_DIR, STORAGE_DIR

import warnings

from data_processing.data_processing import get_no_shuffle_train_loader, get_num_classes
from models.model import load_model
from results.result_processing import get_trace_reduction

warnings.filterwarnings("ignore")


def get_all_losses(model, device, loader):
    model.eval()
    loss_func_sample = torch.nn.CrossEntropyLoss(reduction='none')

    all_losses = []
    with torch.no_grad():
        for inputs, targets, _indices in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(-1).squeeze(-1)
            batch_losses = loss_func_sample(outputs, targets)
            all_losses.append(batch_losses)

    all_losses = torch.cat(all_losses, dim=0)
    return all_losses.tolist()


def prepare_attack(target_model, args, calc_shadow_confs=False):
    device = "cuda" + args.gpu if torch.cuda.is_available() else "cpu"
    attackloader = get_no_shuffle_train_loader(args.dataset, args.arch, args.batchsize,
                                               args.num_workers)
    original_len = len(attackloader.dataset.dataset) if isinstance(attackloader.dataset, Subset) else len(
        attackloader.dataset)
    attack_set = Subset(attackloader.dataset, args.target_trained_on)
    attack_set = DataLoader(attack_set, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    test_idx = [i for i in range(original_len) if i not in args.target_trained_on]
    testset = Subset(attackloader.dataset, test_idx)
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)


    model = load_model(args.arch, get_num_classes(args.dataset)).to(device)

    shadow_train_indices = []  # shadow_models x varied

    dir = os.path.join(MODEL_DIR, args.exp_id)
    model_path = os.path.join(dir, 'shadow_*')

    reference_models = []
    shadow_confs = []

    shadow_test_confs = []

    if calc_shadow_confs:


        ref_model_n = [f for f in glob(model_path)]
        for shadow_idx,f in enumerate(ref_model_n):
            model_path = f
            print("Processing shadow " + str(shadow_idx))
            saves = torch.load(model_path, map_location=device)
            model.load_state_dict(saves['model_state_dict'])
            model.eval()
            reference_models.append(model)

            shadow_confs.append(get_all_losses(model, device, attack_set))
            shadow_test_confs.append(get_all_losses(model, device, testloader))

            shadow_train_indices.append(saves['trained_on_indices'])

        sample_out_confs = {i: [] for i in args.target_trained_on}
        sample_test_confs = {i: [] for i in test_idx}


        for confs_i, i_trained_on in zip(shadow_confs, map(set, shadow_train_indices)):
            mask = np.isin(args.target_trained_on, list(i_trained_on))  # i_trained_on are original/absolute indices
            for sample_idx, conf, in_i_trained_on in zip(args.target_trained_on, confs_i, mask):
                if not in_i_trained_on:
                    sample_out_confs[sample_idx].append(conf)
            np.array(shadow_test_confs).T.tolist()

        for confs_i, i_trained_on in zip(shadow_test_confs, map(set, shadow_train_indices)):
            mask = np.isin(test_idx, list(i_trained_on))  # i_trained_on are original/absolute indices
            for sample_idx, conf, in_i_trained_on in zip(test_idx, confs_i, mask):
                if not in_i_trained_on:
                    sample_test_confs[sample_idx].append(conf)

        np.savez(f"{STORAGE_DIR}/target_model_train_loss_dist",
                 train_loss_dist=sample_out_confs)
        np.savez(f"{STORAGE_DIR}/target_model_test_loss_dist",
                 test_loss_dist=sample_test_confs)


    target_model_confs = get_trace_reduction(exp_id=args.exp_id, target_id=args.target_id, first=-1).iloc[args.target_trained_on]
    target_model_test_confs = get_all_losses(target_model, device, testloader)

    path =f"{STORAGE_DIR}/{args.target_id}"
    os.makedirs(path, exist_ok=True)

    np.savez(f"{STORAGE_DIR}/{args.target_id}/target_model_losses",
             train_losses=target_model_confs, test_losses=target_model_test_confs)

def calculate_loss_threshold(alpha, distribution):
    threshold = np.quantile(distribution, q=alpha, interpolation='lower')
    return threshold


def linear_itp_threshold_func(
        distribution: List[float],
        alpha: List[float],
        signal_min=0,
        signal_max=1000,
        **kwargs
) -> float:
    """
    Function that returns the threshold as the alpha quantile of
    a linear interpolation curve fit over the provided distribution.
    Args:
        distribution: Sequence of values that form the distribution from which
        the threshold is computed. (Here we only consider positive signal values.)
        alpha: Quantile value that will be used to obtain the threshold from the
            distribution.
        signal_min: minimum of all possible signal values, default value is zero
        signal_max: maximum of all possible signal values, default vaule is 1000
    Returns:
        threshold: alpha quantile of the provided distribution.
    """
    distribution = np.append(distribution, signal_min)
    distribution = np.append(distribution, signal_max)
    threshold = np.quantile(distribution, q=alpha, method='linear', **kwargs)
    return threshold


def attack_R(target_model, args, alphas):
    """
           Runs the reference attack on the target model.
    """

    # get train and test loss distributions, loss values
    train_loss_dist_filepath = f"{STORAGE_DIR}/target_model_train_loss_dist.npz"
    test_loss_dist_filepath = f"{STORAGE_DIR}/target_model_test_loss_dist.npz"
    losses_filepath = f"{STORAGE_DIR}/{args.target_id}/target_model_losses.npz"
    if not (os.path.isfile(train_loss_dist_filepath) and os.path.isfile(losses_filepath)\
            and os.path.isfile(test_loss_dist_filepath)):
            prepare_attack(target_model, args, True)

    with np.load(train_loss_dist_filepath, allow_pickle=True) as train_loss_dist_data:
        train_loss_dist = train_loss_dist_data['train_loss_dist'][()]

    with np.load(test_loss_dist_filepath, allow_pickle=True) as test_loss_dist_data:
        test_loss_dist = test_loss_dist_data['test_loss_dist'][()]

    with np.load(losses_filepath, allow_pickle=True) as losses_data:
        train_losses = losses_data['train_losses'][()]
        test_losses = losses_data['test_losses'][()]

    for alpha in alphas:
        print(f"For alpha = {alpha}...")

        # compute threshold for train data
        train_thresholds = []
        train_percs = {}

        threshold_func = calculate_loss_threshold if alpha < 0.001 else linear_itp_threshold_func

        for point_idx, (_,point_loss_dist) in enumerate(train_loss_dist.items()):
            train_thresholds.append(threshold_func(alpha=alpha, distribution=point_loss_dist))
            train_percs[point_idx] = stats.percentileofscore(point_loss_dist, train_losses[point_idx])


        # compute threshold for test data
        test_thresholds = []
        test_percs = {}
        for point_idx, (_,point_loss_dist) in enumerate(test_loss_dist.items()):
                test_thresholds.append(threshold_func(alpha=alpha, distribution=point_loss_dist))
                test_percs[point_idx] = stats.percentileofscore(point_loss_dist, test_losses[point_idx])

        # generate predictions: <= threshold, output '1' (member) else '0' (non-member)
        preds = []
        for (loss, threshold) in zip(train_losses, train_thresholds):
            if loss <= threshold:
                preds.append(1)
            else:
                preds.append(0)

        for (loss, threshold) in zip(test_losses, test_thresholds):
            if loss <= threshold:
                preds.append(1)
            else:
                preds.append(0)

        y_eval = [1] * len(train_losses)
        y_eval.extend([0] * len(test_losses))


        # save attack results
        acc = accuracy_score(y_eval, preds)
        roc_auc = roc_auc_score(y_eval, preds)
        tn, fp, fn, tp = confusion_matrix(y_eval, preds).ravel()
        np.savez(f"{STORAGE_DIR}/{args.target_id}/attack_results_{alpha}",
                 exp_id=args.exp_id,
                 target_id=args.target_id,
                 true_labels=y_eval, preds=preds, alpha=alpha,
                 train_thresholds=train_thresholds,
                 test_thresholds=test_thresholds,
                 acc=acc,
                 roc_auc=roc_auc,
                 tn=tn, fp=fp, tp=tp, fn=fn,
                 train_percs=train_percs,
                 test_percs=test_percs)

        print(
            f"Reference attack performance:\n"
            f"Accuracy = {acc}\n"
            f"ROC AUC Score = {roc_auc}\n"
            f"FPR: {fp / (fp + tn)}\n"
            f"TPR: {tp / (tp + fn)}\n"
            f"TN, FP, FN, TP = {tn, fp, fn, tp}"
        )


def visualize_attack(alphas, args):
    alphas = sorted(alphas)

    tpr_values = []
    fpr_values = []

    for alpha in alphas:
        filepath = f'{STORAGE_DIR}/{args.target_id}/attack_results_{alpha}.npz'
        with np.load(filepath, allow_pickle=True) as data:
            tp = data['tp'][()]
            fp = data['fp'][()]
            tn = data['tn'][()]
            fn = data['fn'][()]
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr_values.append(tpr)
        fpr_values.append(fpr)

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
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_ylim([0.0, 1.1])
    ax.legend(loc='lower right')
    plt.savefig(f'{STORAGE_DIR}/{args.target_id}/tpr_vs_fpr', dpi=250)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str)
    parser.add_argument('--target_id', type=str)# expect to contain some shadow_i of uniform arch
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--alphas', nargs="*", type=str, help="the list of alpha values to run the attack with")
    parser.add_argument('--gpu', default='', type=str, help="select gpu to run e.g. ':0' where multiple are available")

    args = parser.parse_args()
    args.num_workers = 4

    device = "cuda" + args.gpu if torch.cuda.is_available() else "cpu"
    dir = os.path.join(MODEL_DIR, args.exp_id)

    saves = torch.load(os.path.join(dir, args.target_id), map_location=device)

    try:
        args.arch = saves['arch']
        args.dataset = saves['dataset']
        args.trained_on_indices = saves['trained_on_indices']
    except:
        print("WARNING: Could not load all settings.")

    target_model = load_model(args.arch, get_num_classes(args.dataset)).to(device)
    target_model.load_state_dict(saves['model_state_dict'])
    target_model.eval()

    args.target_trained_on = saves['trained_on_indices']

    alphas = args.alphas if args.alphas else np.linspace(0,1,50)

    attack_R(target_model,args, alphas)
    visualize_attack(alphas, args)



if __name__ == '__main__':
    main()