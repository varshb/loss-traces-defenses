import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import torch
from torch.nn import Module
from torch.utils.data import Subset, DataLoader

from data_processing.data_processing import get_no_shuffle_train_loader, get_num_classes
from models.model import load_model

# LOCAL_DIR = '/home/joseph/rds/home/loss_traces' # path to this folder
# # paths to store stuff...
# STORAGE_DIR = '/home/joseph/rds/home/'
# MODEL_DIR = '/home/joseph/rds/home/trained_models/'
# DATA_DIR = '/home/joseph/rds/ephemeral/data/'

# LOCAL_DIR = '/data_2/euodia/loss_traces' # path to this folder
# # paths to store stuff...
# STORAGE_DIR = '/data_2/euodia/'

# MY_STORAGE_DIR = '/data_2/euodia/'
MY_SECONDARY_STORAGE_DIR = '/home/euodia/rds/home/'
MODEL_DIR = '/home/joseph/rds/home/trained_models/'
# DATA_DIR = '/data_2/euodia/data'

@dataclass
class AttackConfig:
    exp_id: str
    target_id: str
    checkpoint: Optional[str]
    arch: str
    dataset: str
    attack: str
    augment: bool =  False
    batchsize: int = 500
    num_workers: int = 8
    gpu: str = ''


class ModelConfidenceExtractor:
    def get_logits(self, model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader,  metrics=["logits"])[1]
    def get_scaled_logits(self, model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader, metrics=["scaled_logits"])[2]
    def get_target_logits(self, model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader, metrics=["target_logits"])[3]
    def get_losses(self,  model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader, metrics=["losses"])[0]
    # def get_all_metrics(self, model: Module, device: torch.device, loader: DataLoader):
    #     return self._get_metrics(model, device, loader, metrics=["losses", "logits", "scaled_logits"])

    @staticmethod
    def _get_metrics(model: Module, device: torch.device, loader: DataLoader, metrics=["losses", "logits", "scaled_logits"]) -> List[float]:
        """Extract metrics from model predictions."""
        model.eval()

        losses = []
        logits = []
        scaled_logits = []
        target_logits = []

        with torch.no_grad():
            for inputs, targets, _indices in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze(-1).squeeze(-1)
                pred_confs, _ = torch.max(outputs, dim=1)
                target_confs = outputs[torch.arange(outputs.shape[0]), targets]

                if "losses" in metrics:
                    loss_func_sample = torch.nn.CrossEntropyLoss(reduction='none')
                    m = loss_func_sample(outputs, targets)
                    losses.extend(m.tolist())

                if "logits" in metrics:
                    logits.extend(pred_confs.tolist())
                #
                if "target_logits" in metrics:
                    target_logits.extend(target_confs.tolist())


                if "scaled_logits" in metrics:
                    outputs[torch.arange(outputs.shape[0]), targets] = float('-inf')
                    pred_confs, _ = torch.max(outputs, dim=1)
                    m = target_confs - pred_confs
                    scaled_logits.extend(m.tolist())

        return losses, logits, scaled_logits, target_logits

class MembershipInferenceAttack:
    def __init__(self, config: AttackConfig):
        self.config = config
        self.device = torch.device(f"cuda{config.gpu}" if torch.cuda.is_available() else "cpu")
        self.model_dir = self._get_model_directory()
        self.model, self.attack_loaders = self._initialize_model_and_data()
        self.extractor = ModelConfidenceExtractor()
        self.dataset_size = (len(self.attack_loaders[0].dataset.dataset)
                        if isinstance(self.attack_loaders[0].dataset, Subset)
                        else len(self.attack_loaders[0].dataset))

    def _get_model_directory(self) -> Path:
        dir_path = Path(MODEL_DIR) / self.config.exp_id
        if self.config.checkpoint:
            dir_path = dir_path / f'checkpoint_before_{self.config.checkpoint}'
        return dir_path

    def _initialize_model_and_data(self) -> Tuple[Module, DataLoader]:
        attack_loaders = [
            get_no_shuffle_train_loader(
                self.config.dataset,
                self.config.arch,
                self.config.batchsize,
                self.config.num_workers
            )
        ]

        if self.config.augment:
            attack_loaders.append(
                get_no_shuffle_train_loader(
                    self.config.dataset,
                    self.config.arch,
                    self.config.batchsize,
                    self.config.num_workers,
                    mirror_all=True)
            )

        model = load_model(
            self.config.arch,
            get_num_classes(self.config.dataset)
        ).to(self.device)

        return model, attack_loaders

    def _collect_shadow_model_data(self, metrics=["losses", "logits", "scaled_logits"]) -> Tuple[
        Dict[str, List[List[float]]], List[List[int]]]:
        """Collect all metrics from shadow models in a single pass.

        Args:
            metrics: List of metrics to collect (defaults to all)

        Returns:
            Tuple containing:
            - Dictionary of metric arrays for each metric type
            - List of training indices for each shadow model
        """
        # Initialize metric storage for each augmentation
        shadow_metrics = {
            metric: [[] for _ in self.attack_loaders]
            for metric in metrics
        }
        shadow_train_indices = []

        shadow_idx = 0
        while True:
            model_path = self.model_dir / f'shadow_{shadow_idx}'
            if not model_path.is_file():
                break

            saves = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(saves['model_state_dict'])

            for i, loader in enumerate(self.attack_loaders):
                res = self.extractor._get_metrics(
                    self.model,
                    self.device,
                    loader,
                    metrics=metrics
                )

                # Store each metric type
                metric_values = res
                for metric_name, value in zip(metrics, metric_values):
                    if value:  # Only store non-empty results
                        shadow_metrics[metric_name][i].append(value)

            shadow_train_indices.append(saves['trained_on_indices'])
            print(f"Computed metrics for shadow {shadow_idx}")
            shadow_idx += 1

        # Transpose each metric array
        for metric in metrics:
            shadow_metrics[metric] = np.array(shadow_metrics[metric]).transpose(1, 2, 0).tolist()

        return shadow_metrics, shadow_train_indices

    def _compute_statistics(self, shadow_confs: List[List[float]],
                            shadow_train_indices: List[List[int]]) -> Dict:
        """Compute statistical measures for attack analysis."""
        all_indices = list(range(self.dataset_size))

        sample_in_confs = {i: [] for i in all_indices}
        sample_out_confs = {i: [] for i in all_indices}

        # Categorize confidences
        for confs, trained_indices in zip(shadow_confs, map(set, shadow_train_indices)):
            mask = np.isin(all_indices, list(trained_indices))
            for idx, conf, in_trained in zip(all_indices, confs, mask):
                target_dict = sample_in_confs if in_trained else sample_out_confs
                target_dict[idx].append(conf)

        if self.config.augment:
            f = lambda x: np.cov(x, rowvar=False)
        else:
            f = np.var

        # Compute statistics
        if len(shadow_confs) >= 64:
            in_var = [f(confs) for confs in sample_in_confs.values()]
            out_var = [f(confs) for confs in sample_out_confs.values()]
        else:
            global_in_var = f([c for confs in sample_in_confs.values() for c in confs])
            global_out_var = f([c for confs in sample_out_confs.values() for c in confs])
            in_var = [global_in_var] * self.dataset_size
            out_var = [global_out_var] * self.dataset_size

        in_means = [np.mean(confs, axis=0) for confs in sample_in_confs.values()]
        out_means = [np.mean(confs, axis=0) for confs in sample_out_confs.values()]

        return {
            'in_conf': sample_in_confs,
            'out_conf': sample_out_confs,
            'in_var': in_var,
            'out_var': out_var,
            'in_means': in_means,
            'out_means': out_means
        }

    def save_intermediate_results(self, stats: Dict, output_dir: str = 'logits_intermediate'):
        """Save intermediate statistical results."""
        save_dir = Path(MY_SECONDARY_STORAGE_DIR) / output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = self.config.exp_id
        if self.config.checkpoint:
            file_name += f'_checkpoint_after_{self.config.checkpoint}'

        torch.save(stats, os.path.join(save_dir, f'{file_name}.pt'))

    def compute_intermediate_results(self):
        """Compute and save intermediate statistical results in a single pass."""
        print("Computing all metrics...")

        start_time = time.time()

        # Collect all metrics in one pass
        metrics = ["losses", "logits", "scaled_logits", "target_logits"]
        shadow_metrics, shadow_indices = self._collect_shadow_model_data(metrics)

        # Compute and save statistics for each metric type
        for metric_name, metric_data in shadow_metrics.items():
            print(f"Computing statistics for {metric_name}...")
            stats = self._compute_statistics(metric_data, shadow_indices)
            self.save_intermediate_results(stats, output_dir=f'{metric_name}_intermediate')

        end_time = time.time()

        print(f"Performance Timings:")
        print(f"Total computation time: {end_time - start_time:.2f}s")
        print(f"Finished stats")

    def _load_intermediate_stats(self, metric) -> pd.DataFrame:
        """Load intermediate statistical results from saved files.

        Returns:
            pd.DataFrame containing the previously computed statistics

        Raises:
            FileNotFoundError: If intermediate results file doesn't exist
        """
        stats_dir = Path(MY_SECONDARY_STORAGE_DIR) / f'{metric}_intermediate'

        file_name = self.config.exp_id
        if self.config.checkpoint:
            file_name += f'_checkpoint_after_{self.config.checkpoint}'

        stats_path = stats_dir / f'{file_name}.pt'

        if not stats_path.exists():
            raise FileNotFoundError(f"No intermediate results found at {stats_path}")

        return torch.load(stats_path)

    def _save_attack_results(self, scores: List[float], target_trained_on: List[int], output_dir: str, **kwargs):
        """Save attack results in CSV format with membership indicators."""
        # Create membership boolean array
        bools = [False] * len(self.attack_loaders[0].dataset)
        for i, (_, _, idx) in enumerate(self.attack_loaders[0].dataset):
            if idx in target_trained_on:
                bools[i] = True

        # Get original dataset length
        original_len = (len(self.attack_loaders[0].dataset.dataset)
                        if isinstance(self.attack_loaders[0].dataset, Subset)
                        else len(self.attack_loaders[0].dataset))

        # Generate all indices
        all_indices = list(range(original_len))

        # Create save directory
        save_dir = Path(MY_SECONDARY_STORAGE_DIR) / f"{output_dir}_scores"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate result filename
        result_name = (f'{self.config.exp_id}_{self.config.target_id}' if self.config.checkpoint is None
                       else f'{self.config.exp_id}_checkpoint_before_{self.config.checkpoint}_{self.config.target_id}')

        # Handle file collisions
        fullpath = save_dir / result_name
        if fullpath.exists():
            print("RESULTS EXIST BUT NOT OVERWRITING", file=sys.stderr)
        while fullpath.exists():
            fullpath = Path(str(fullpath) + "_")

        # Save results
        results = pd.DataFrame({
            f'{output_dir.rstrip("s")}_score': scores,
            'target_trained_on': bools,
            'og_idx': all_indices
        }).set_index('og_idx')

        if kwargs:
            extra =  pd.DataFrame(kwargs)
            results = pd.concat([results, extra], axis=1)


        results.to_csv(fullpath)


class LiRAAttack(MembershipInferenceAttack):
    def run(self):
        """Execute LiRA (Likelihood Ratio Attack)."""
        # Load target model
        saves = torch.load(self.model_dir / self.config.target_id, map_location=self.device)
        self.model.load_state_dict(saves['model_state_dict'])
        target_indices = saves['trained_on_indices']

        # Get target model confidences
        # target_confs = self.extractor.get_scaled_logits(self.model, self.device, self.attack_loaders)
        target_confs = [self.extractor.get_scaled_logits(self.model, self.device, loader) for loader in self.attack_loaders]
        target_confs = np.array(target_confs).T

        # Load intermediate results
        stats_df = self._load_intermediate_stats("scaled_logits")

        # Compute LiRA scores
        lira_scores = self._compute_lira_scores(target_confs, stats_df)

        # Save results
        self._save_attack_results(lira_scores, target_indices, 'lira')

    def _compute_lira_scores(self, target_confs: List[float], stats_df: pd.DataFrame) -> List[float]:
        """Compute LiRA scores using statistical analysis."""

        scores = []
        if self.config.augment:
            for i, conf in enumerate(target_confs):
                l = scipy.stats.multivariate_normal.pdf(conf, mean=stats_df['in_means'][i],
                                                        cov=stats_df['in_var'][i],  allow_singular=True) + 1e-64  # TODO
                r = scipy.stats.multivariate_normal.pdf(conf, mean=stats_df['out_means'][i], cov=stats_df['out_var'][i],  allow_singular=True) + 1e-64
                score = l / r
                scores.append(score)
        else:
            stats_df['in_std'] = np.sqrt(stats_df['in_var'])
            stats_df['out_std'] = np.sqrt(stats_df['out_var'])
            for i, conf in enumerate(target_confs):
                l = scipy.stats.norm.pdf(conf, loc=stats_df['in_means'][i], scale=stats_df['in_std'][i],  allow_singular=True) + 1e-64  # TODO
                r = scipy.stats.norm.pdf(conf, loc=stats_df['out_means'][i], scale=stats_df['out_std'][i],  allow_singular=True) + 1e-64
                score = l / r
                scores.append(score)

        return scores



class RMIAAttack(MembershipInferenceAttack):
    def run(self, gamma: float = 2.0):
        """Execute RMIA (Relative Membership Inference Attack)."""
        # Load target model
        saves = torch.load(self.model_dir / self.config.target_id, map_location=self.device)
        self.model.load_state_dict(saves['model_state_dict'])
        target_indices = saves['trained_on_indices']

        # Get confidences and compute ratios
        target_confs = self.extractor.get_logits(self.model, self.device, self.attack_loaders[0])
        stats_df = self._load_intermediate_stats("logits")

        # Compute RMIA scores
        rmia_scores = self._compute_rmia_scores(target_confs, target_indices, stats_df, gamma)

        # Save results
        self._save_attack_results(rmia_scores, target_indices, f'rmia_{gamma}')

    def _compute_rmia_scores(self, target_confs: List[float], target_indices: List[int],
                             stats_df: pd.DataFrame, gamma: float) -> List[float]:
        """Compute RMIA scores using relative likelihood analysis."""
        test_indices = list(set(range(self.dataset_size)) - set(target_indices))
        out_conf = [[x[0] for x in dists] for k, dists in stats_df['out_conf'].items()]
        in_conf = [[x[0] for x in dists] for k, dists in stats_df['in_conf'].items()]

        ratio_z = [target_confs[z] / np.mean(out_conf[z]) for z in test_indices]

        scores = []

        for i, conf in enumerate(target_confs):
            pr_x = (np.sum(in_conf[i]) + np.sum(out_conf[i])) / (2 * len(out_conf[i]))
            ratio_x = conf / pr_x
            scores.append(sum((ratio_x / r_z) > gamma for r_z in ratio_z))

        return scores

class AttackR(MembershipInferenceAttack):
    # np.linspace(0, 1, 100)
    def run(self, alphas=np.logspace(-5, 0, 100)):
        """Execute RMIA (Relative Membership Inference Attack)."""
        # Load target model
        saves = torch.load(self.model_dir / self.config.target_id, map_location=self.device)
        self.model.load_state_dict(saves['model_state_dict'])
        target_indices = saves['trained_on_indices']

        # Get confidences and compute ratios
        target_confs = self.extractor.get_losses(self.model, self.device, self.attack_loaders[0])
        stats_df = self._load_intermediate_stats("losses")

        # Save results
        for alpha in alphas:
            # Compute AttackR scores
            attackr_scores, thresholds, preds = self._compute_attackr_scores(target_confs, target_indices, stats_df, alpha)

        self._save_attack_results(attackr_scores, target_indices, f'attackr_{alpha}', thresholds=thresholds, preds=preds)



    @staticmethod
    def calculate_loss_threshold(alpha, distribution):
        threshold = np.quantile(distribution, q=alpha, interpolation='lower')
        return threshold

    @staticmethod
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

    def _compute_attackr_scores(self, target_confs: List[float], target_indices: List[int],
                             stats_df: pd.DataFrame, alpha: List[float]) -> List[float]:
        """Compute RMIA scores using relative likelihood analysis."""
        # for alpha in alphas:
        train_thresholds = []
        train_percs = []

        threshold_func = self.calculate_loss_threshold if alpha < 0.001 else self.linear_itp_threshold_func

        out_conf = [[x[0] for x in dists] for k,dists in stats_df['out_conf'].items()]

        for point_idx, point_loss_dist in enumerate(out_conf):
            train_thresholds.append(threshold_func(alpha=alpha, distribution=point_loss_dist))
            train_percs.append(stats.percentileofscore(point_loss_dist, target_confs[point_idx]))

        preds = []
        for loss, threshold in zip(target_confs, train_thresholds):
            if loss <= threshold:
                preds.append(1)
            else:
                preds.append(0)

        return train_percs, train_thresholds, preds



def parse_args() -> AttackConfig:
    """Parse command line arguments and return attack configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batchsize', type=int, default=500)
    parser.add_argument('--target_id', default='target', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--attack', type=str, required=True, choices=['LiRA', 'RMIA', 'AttackR'])

    args = parser.parse_args()

    # Load saved configurations if available
    model_dir = Path(MODEL_DIR) / args.exp_id
    if args.checkpoint:
        model_dir = model_dir / f'checkpoint_before_{args.checkpoint}'

    try:
        saves = torch.load(model_dir / 'shadow_0')
        args.arch = args.arch or saves['arch']
        args.dataset = args.dataset or saves['dataset']
        args.augment = saves['hyperparameters']['augment']
    except Exception as e:
        print(f"Warning: Could not load all settings: {e}")

    return AttackConfig(
        exp_id=args.exp_id,
        target_id=args.target_id,
        checkpoint=args.checkpoint,
        arch=args.arch,
        dataset=args.dataset,
        attack=args.attack,
        ##TODO: Put back augmentation
        augment=args.augment,
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        gpu=args.gpu
    )


def main():
    config = parse_args()

    # Initialize and run appropriate attack
    if config.attack == 'LiRA':
        attack = LiRAAttack(config)
    elif config.attack == "RMIA":
        attack = RMIAAttack(config)
    else:  # RMIA
        attack = AttackR(config)

    # Compute intermediate results if they don't exist
    attack.compute_intermediate_results()
    attack.run()
    # attack.run(alphas=np.logspace(-5, 0, 100))
    # attack.run(alphas=np.linspace(0, 1, 100))


if __name__ == "__main__":
    main()