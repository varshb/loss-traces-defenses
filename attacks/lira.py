import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import scipy
import torch
from torch.nn import Module
from torch.utils.data import Subset, DataLoader

from config import MODEL_DIR, STORAGE_DIR
from data_processing.data_processing import get_no_shuffle_train_loader, get_num_classes
from models.model import load_model


def get_scaled_logits(model: Module, device: torch.device, loader: DataLoader) -> list:
    model.eval()
    model_confs = []

    with torch.no_grad():
        for inputs, targets, _indices in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs).squeeze(-1).squeeze(-1)

            target_confs = outputs[torch.arange(outputs.shape[0]), targets]
            outputs[torch.arange(outputs.shape[0]), targets] = float('-inf')
            pred_confs, _ = torch.max(outputs, dim=1)
            batch_confs = target_confs - pred_confs

            model_confs.extend(batch_confs.tolist())

    return model_confs


class LiRA:
    def __init__(self, args):
        """
        Initialize the Membership Inference Attack analysis.

        Args:
            exp_id (str): Experiment identifier
            checkpoint (Optional[str]): Checkpoint identifier
        """
        self.start_time = time.time()
        self.exp_id = args.exp_id
        self.target_id = args.target_id
        self.checkpoint = args.checkpoint

        self.args = args
        self.device = self._setup_device()

        # Model and data loading
        self.model_dir = self._get_model_directory()
        self.model, self.attack_loaders = self._load_model_and_data()

    # def _parse_arguments(self) -> argparse.Namespace:
    #     """Parse command line arguments."""
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--exp_id', type=str, required=True)
    #     parser.add_argument('--checkpoint', default=None)
    #     parser.add_argument('--arch', type=str)
    #     parser.add_argument('--dataset', type=str)
    #     parser.add_argument('--batchsize', type=int, default=500)
    #     parser.add_argument('--target_id', default='target', type=str, help='name of model to attack')
    #     parser.add_argument('--num_workers', type=int, default=8)
    #     parser.add_argument('--gpu', default='', type=str, help="select gpu to run e.g. ':0'")
    #
    #     args = parser.parse_args()
    #
    #     # Load saved configurations if possible
    #     try:
    #         saves = torch.load(os.path.join(MODEL_DIR, args.exp_id, 'shadow_0'))
    #         args.arch = saves.get('arch', args.arch)
    #         args.dataset = saves.get('dataset', args.dataset)
    #     except Exception as e:
    #         print(f"WARNING: Could not load all settings: {e}")
    #
    #     return args

    def _setup_device(self) -> str:
        """Set up the computation device."""
        return f"cuda{self.args.gpu}" if torch.cuda.is_available() else "cpu"

    def _get_model_directory(self) -> str:
        """Determine the model directory path."""
        dir_path = os.path.join(MODEL_DIR, self.exp_id)
        if self.checkpoint is not None:
            dir_path = os.path.join(dir_path, f'checkpoint_before_{self.checkpoint}')
        return dir_path

    def _load_model_and_data(self):
        """Load the model and attack data loader."""
        # Create attack loader
        attack_loaders = [
            get_no_shuffle_train_loader(
                self.args.dataset,
                self.args.arch,
                self.args.batchsize,
                self.args.num_workers
            )
        ]

        if self.args.augment:
            attack_loaders.append(
                get_no_shuffle_train_loader(
                    self.args.dataset, 
                    self.args.arch, 
                    self.args.batchsize, 
                    self.args.num_workers, 
                    mirror_all=True)
            )

        # Load model
        model = load_model(self.args.arch, get_num_classes(self.args.dataset)).to(self.device)

        return model, attack_loaders

    def _collect_model_confidences(self) -> tuple:
        """
        Collect confidences from shadow models.

        Returns:
            tuple: shadow model confidences and training indices
        """
        shadow_confs = [[] for _ in self.attack_loaders] # augmentations x shadow_models x samples
        shadow_train_indices = []  # shadow_models x varied

        shadow_idx = 0
        while True:
            model_path = os.path.join(self.model_dir, f'shadow_{shadow_idx}')
            if not os.path.isfile(model_path):
                break

            saves = torch.load(model_path, map_location=self.device)

            self.model.load_state_dict(saves['model_state_dict'])
            self.model.eval()

            # Collect model confidences
            for i, loader in enumerate(self.attack_loaders):
                shadow_confs[i].append(get_scaled_logits(self.model, self.device, loader))

            shadow_train_indices.append(saves['trained_on_indices'])

            shadow_idx += 1

        shadow_confs = np.array(shadow_confs).transpose(1,2,0).tolist()

        return shadow_confs, shadow_train_indices

    def _compute_sample_confidences(self, shadow_confs, shadow_train_indices) -> tuple:
        """
        Compute sample confidences for in-training and out-of-training samples.

        Returns:
            tuple: in-variance, out-variance, in-means, out-means
        """
        original_len = (len(self.attack_loaders[0].dataset.dataset)
                        if isinstance(self.attack_loaders[0].dataset, Subset)
                        else len(self.attack_loaders[0].dataset))

        # Get all valid indices
        all_indices = [i for i in range(original_len)]

        # Initialize confidence tracking dictionaries
        sample_in_confs = {i: [] for i in all_indices}
        sample_out_confs = {i: [] for i in all_indices}

        # Categorize sample confidences
        for confs_i, i_trained_on in zip(shadow_confs, map(set, shadow_train_indices)):
            mask = np.isin(all_indices, list(i_trained_on))
            for sample_idx, conf, in_i_trained_on in zip(all_indices, confs_i, mask):
                if in_i_trained_on:
                    sample_in_confs[sample_idx].append(conf)
                else:
                    sample_out_confs[sample_idx].append(conf)

        if self.args.augment:
            f = lambda x: np.cov(x, rowvar=False)
        else:
            f = np.var

        # Compute variance (local vs global)
        if len(shadow_confs) >= 64:
            in_var = [f(confs) for confs in sample_in_confs.values()]
            out_var = [f(confs) for confs in sample_out_confs.values()]
        else:
            in_var = [f([conf for confs in sample_in_confs.values() for conf in confs])] * len(all_indices)
            out_var = [f([conf for confs in sample_out_confs.values() for conf in confs])] * len(all_indices)

        # Compute means
        in_means = [np.mean(confs, axis=0) for confs in sample_in_confs.values()]
        out_means = [np.mean(confs, axis=0) for confs in sample_out_confs.values()]

        return in_var, out_var, in_means, out_means

    def _save_results(self, in_var, out_var, in_means, out_means):
        """
        Save the computed results to a Parquet file.
        """
        dir_path = os.path.join(STORAGE_DIR, 'lira_scores/intermediate')
        os.makedirs(dir_path, exist_ok=True)

        data = {
            'in_var': in_var,
            'out_var': out_var,
            'in_means': in_means,
            'out_means': out_means
        }

        file_name = self.exp_id
        if self.checkpoint is not None:
            file_name += f'_checkpoint_after_{self.checkpoint}'

        torch.save(data, os.path.join(dir_path, f'{file_name}.pt'))

    def get_and_store_intermediate(self):
        """
        Execute the full membership inference attack pipeline.
        """
        print(f"Preparing to perform LiRA")

        # Performance tracking
        start_time = time.time()

        # Collect model confidences
        shadow_confs, shadow_train_indices = self._collect_model_confidences()
        post_logits_time = time.time()

        # Compute sample confidences
        in_var, out_var, in_means, out_means = self._compute_sample_confidences(
            shadow_confs,
            shadow_train_indices
        )
        numerical_compute_time = time.time()

        # Save results
        self._save_results(in_var, out_var, in_means, out_means)

        # Print performance timings
        print("----Timings----")
        print(f"Init/loading: {post_logits_time - start_time:.2f}")
        print(f"Iterating/logits: {numerical_compute_time - post_logits_time:.2f}")
        print(f"Numerical compute: {time.time() - numerical_compute_time:.2f}")

    def run_attack(self):
        path = self._get_model_directory()
        saves = torch.load(os.path.join(path, self.target_id), map_location=self.device)

        self.model.load_state_dict(saves['model_state_dict'])
        target_trained_on = saves['trained_on_indices']
        self.model.eval()

        target_model_confs = [get_scaled_logits(self.model, self.device, loader) for loader in self.attack_loaders]
        target_model_confs = np.array(target_model_confs).T

        file_name = self.exp_id
        if self.checkpoint is not None:
            file_name += f'_checkpoint_after_{self.checkpoint}'

        df = torch.load(os.path.join(STORAGE_DIR, 'lira_scores/intermediate', file_name + '.pt'))

        lira_scores = []
        if self.args.augment:
            for i, conf in enumerate(target_model_confs):
                l = scipy.stats.multivariate_normal.pdf(conf, mean=df['in_means'][i], cov=df['in_var'][i]) + 1e-64 # TODO
                r = scipy.stats.multivariate_normal.pdf(conf, mean=df['out_means'][i], cov=df['out_var'][i]) + 1e-64
                score = l / r
                lira_scores.append(score)
        else:
            df['in_std'] = np.sqrt(df['in_var'])
            df['out_std'] = np.sqrt(df['out_var'])

            for i, conf in enumerate(target_model_confs):
                l = scipy.stats.norm.pdf(conf, loc=df['in_means'][i], scale=df['in_std'][i]) + 1e-64 # TODO
                r = scipy.stats.norm.pdf(conf, loc=df['out_means'][i], scale=df['out_std'][i]) + 1e-64
                score = l / r
                lira_scores.append(score)

        bools = [False] * len(self.attack_loaders[0].dataset)
        for i, (_, _, idx) in enumerate(self.attack_loaders[0].dataset):
            if idx in target_trained_on:
                bools[i] = True

        original_len = len(self.attack_loaders[0].dataset.dataset) if isinstance(self.attack_loaders[0].dataset, Subset) else len(
            self.attack_loaders[0].dataset)

        all_indices = [i for i in range(original_len)]

        dir = os.path.join(STORAGE_DIR, 'lira_scores')
        os.makedirs(dir, exist_ok=True)
        result_name = f'{self.exp_id}_{self.target_id}' if self.checkpoint is None else f'{self.exp_id}_checkpoint_before_{self.checkpoint}_{self.target_id}'
        fullpath = os.path.join(dir, result_name)
        if os.path.exists(fullpath):
            print("RESULTS EXIST BUT NOT OVERWRITING", file=sys.stderr)
        while os.path.exists(fullpath):
            fullpath += "_"
        pd.DataFrame({'lira_score' : lira_scores, 'target_trained_on' : bools, 'og_idx': all_indices}).set_index('og_idx').to_csv(fullpath)



def main():
    # Parse exp_id and optional checkpoint from command line
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batchsize', type=int, default=500)
    parser.add_argument('--target_id', default='target', type=str, help='name of model to attack')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', default='', type=str, help="select gpu to run e.g. ':0'")

    args = parser.parse_args()

    # Load saved configurations if possible
    dir = os.path.join(MODEL_DIR, args.exp_id)
    if args.checkpoint is not None:
        dir = os.path.join(dir, f'checkpoint_before_{args.checkpoint}')
    saves = torch.load(os.path.join(dir, 'shadow_0'))

    try:
        args.arch = saves['arch']
        args.dataset = saves['dataset']
        args.augment = saves['hyperparameters']['augment']
    except:
        print("WARNING: Could not load all settings.")

    # Run the membership inference attack
    attack = LiRA(args)
    attack.get_and_store_intermediate()
    attack.run_attack()


if __name__ == "__main__":
    main()