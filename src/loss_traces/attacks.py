import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys

import numpy as np
import opacus
import pandas as pd
import scipy
from opacus import GradSampleModule
from opacus.validators import ModuleValidator
from scipy import stats
import torch
import pickle
from sklearn import metrics
from torch.nn import Module
from torch.utils.data import Subset, DataLoader

from loss_traces.config import MODEL_DIR, STORAGE_DIR
from loss_traces.data_processing.data_processing import get_no_shuffle_train_loader, get_num_classes
from loss_traces.main import set_seed
from loss_traces.models.model import load_model

@dataclass
class AttackConfig:
    exp_id: str
    target_id: str
    checkpoint: Optional[str]
    arch: str
    dataset: str
    attack: str
    n_shadows: Optional[int] = None
    shadow_seed: Optional[int] = None
    offline: Optional[str] = None
    augment: bool =  True
    batchsize: int = 500
    num_workers: int = 8
    gpu: str = ''
    is_dp: bool = False
    layer: int = 0  # Number of layers to use for the attack, default is 0 (all layers)
    layer_folder: str = None # Folder to retrieve layer-indices from, if applicable

class ModelConfidenceExtractor:
    def get_logits(self, model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader,  metrics=["logits"])[1]
    def get_scaled_logits(self, model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader, metrics=["scaled_logits"])[2]
    def get_target_logits(self, model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader, metrics=["target_logits"])[3]
    def get_losses(self,  model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader, metrics=["losses"])[0]


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
        print(f"Dataset size: {self.dataset_size}")

    def _get_model_directory(self) -> Path:
        dir_path = Path(MODEL_DIR) / self.config.exp_id
        if self.config.checkpoint:
            dir_path = dir_path / f'checkpoint_before_{self.config.checkpoint}'
        return dir_path

    def _initialize_model_and_data(self) -> Tuple[Module, DataLoader]:
        if self.config.layer > 0:
            print(f"Using {self.config.layer} layers for attack")
            save_path = f"{STORAGE_DIR}/layer_target_indices/{self.config.layer_folder}/layer_{self.config.layer - 1}_full_safe.pkl"
            print(f"Loading safe indices from {save_path}")
            with open(save_path, "rb") as f:
                safe_indices = pickle.load(f)
            self.safe_indices = list(safe_indices['og_idx'])
            print(f"Safe indices loaded: {len(self.safe_indices)}")
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
                    mirror_all=True    
            )
            )

        model = load_model(
            self.config.arch,
            get_num_classes(self.config.dataset)
        ).to(self.device)

        return model, attack_loaders
    
    def _is_dp_model_needed(self, hyperparameters: Dict) -> bool:
        """Determine if the model needs DP compatibility."""
        return (hyperparameters.get("private") or
                hyperparameters.get("clip_norm") is not None or
                hyperparameters.get("noise_multiplier") is not None)
    
    def _convert_to_dp_model(self, model: Module) -> Module:
        """Convert a model to a DP-compatible model."""

        if not isinstance(model, opacus.GradSampleModule):
            model = ModuleValidator.fix(model)
            model = GradSampleModule(model)

        return model
        
    def _load_model_with_dp_check(self, model_path, strict=True):
        """Load a model with DP compatibility check.
        
        Args:
            model_path: Path to the saved model
            strict: Whether to use strict loading for state_dict
            
        Returns:
            Tuple containing:
            - Model state after loading
            - Training indices from the saved model
        """
        saves = torch.load(model_path, map_location=self.device, weights_only=False)
        print(model_path)
        # Check if DP model is needed
        if self._is_dp_model_needed(saves['hyperparameters']):
            self.model = self._convert_to_dp_model(self.model)
            
        # Load state dictionary
        self.model.load_state_dict(saves['model_state_dict'], strict=strict)
        
        return self.model, saves['trained_on_indices']

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

            # Load model with DP check
            self.model, indices = self._load_model_with_dp_check(model_path)
            shadow_train_indices.append(indices)

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
                if in_trained:
                    sample_in_confs[idx].append(conf)
                elif self.config.layer > 0 and idx in self.safe_indices:
                    sample_out_confs[idx].append(conf)
                elif self.config.layer == 0:
                    sample_out_confs[idx].append(conf)

        in_var, out_var, in_means, out_means = self.compute_metrics(sample_in_confs, sample_out_confs, len(shadow_confs))
        return {
            'in_conf': sample_in_confs,
            'out_conf': sample_out_confs,
            'in_var': in_var,
            'out_var': out_var,
            'in_means': in_means,
            'out_means': out_means
        }

    def compute_metrics(self, sample_in_confs, sample_out_confs, n_shadows):

        if self.config.augment:
            f = lambda x: np.cov(x, rowvar=False) 
        else:
            f = np.var

        # Compute statistics
        if n_shadows >= 68:
            in_var = [f(confs) for confs in sample_in_confs.values() if len(confs) > 0]
            out_var = [f(confs) for confs in sample_out_confs.values() if len(confs) > 0]
        else:
            global_in_var = f([c for confs in sample_in_confs.values() for c in confs])
            global_out_var = f([c for confs in sample_out_confs.values() for c in confs])
            in_var = [global_in_var] * self.dataset_size
            out_var = [global_out_var] * self.dataset_size

        in_means = [np.mean(confs, axis=0) if len(confs) > 0 else np.nan for confs in sample_in_confs.values()]
        out_means = [np.mean(confs, axis=0) if len(confs) > 0 else np.nan for confs in sample_out_confs.values()]
       
        return in_var, out_var, in_means, out_means

    def select_subset_shadow_metrics(self, stats_df):
        if self.config.n_shadows:
            if self.config.shadow_seed:
                np.random.seed(self.config.shadow_seed)
            selected_idx = np.random.choice(range(len(stats_df['in_conf'][0])), self.config.n_shadows, replace=False)
            in_conf = {key: [stats_df['in_conf'][key][idx] for idx in selected_idx] if len(stats_df['in_conf'][key]) > 0 else [] for key in stats_df['in_conf']}
            out_conf = {key: [stats_df['out_conf'][key][idx] for idx in selected_idx] if len(stats_df['out_conf'][key]) > 0 else [] for key in stats_df['out_conf']}

            if self.config.offline:
                n = self.config.n_shadows
            else:
                n = self.config.n_shadows*2
            in_var, out_var, in_means, out_means = self.compute_metrics(in_conf, out_conf, n)
        else:
            in_var, out_var, in_means, out_means = stats_df['in_var'], stats_df['out_var'], stats_df['in_means'], stats_df['out_means']
        return in_var, out_var, in_means, out_means
        
    def save_intermediate_results(self, stats: Dict, output_dir: str = 'logits_intermediate'):
        """Save intermediate statistical results."""
        save_dir = Path(STORAGE_DIR) / output_dir
        if self.config.checkpoint:
            save_dir = save_dir / f'checkpoint_before_{self.config.checkpoint}'
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = self.config.exp_id

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
        stats_dir = Path(STORAGE_DIR) / f'{metric}_intermediate'

        file_name = self.config.exp_id
        if self.config.checkpoint:
            stats_dir = stats_dir / f'checkpoint_before_{self.config.checkpoint}'

        stats_path = stats_dir / f'{file_name}.pt'

        # Compute intermediate results if they don't exist
        if not stats_path.exists():
            print(f"No intermediate results found at {stats_path}. Computing intermediate results...")
            self.compute_intermediate_results()

        return torch.load(stats_path, weights_only=False)

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
        save_dir = Path(STORAGE_DIR) / f"{output_dir}_scores"

        if self.config.checkpoint:
            save_dir = save_dir / f'checkpoint_before_{self.config.checkpoint}'
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate result filename
        result_name = f'{self.config.exp_id}_{self.config.target_id}'

        # Handle file collisions
        fullpath = save_dir / result_name

        # Save results
        results = pd.DataFrame({
            f'{output_dir.rstrip("s")}_score': scores,
            'target_trained_on': bools,
            'og_idx': all_indices
        }).set_index('og_idx')

        if self.config.layer > 0:
            results = results[results.index.isin(self.safe_indices)]
            
        if kwargs:
            extra =  pd.DataFrame(kwargs)
            results = pd.concat([results, extra], axis=1)

        if self.config.offline:
            fullpath = Path(str(fullpath) + f"_offline")
        if self.config.n_shadows:
            fullpath = Path(str(fullpath) + f"_{self.config.n_shadows}")

        if fullpath.exists():
            print("RESULTS EXIST BUT NOT OVERWRITING", file=sys.stderr)
        while fullpath.exists():
            fullpath = Path(str(fullpath) + "_")

        print("Attack AUC:", metrics.roc_auc_score(results['target_trained_on'], results[f'{output_dir.rstrip("s")}_score']))
        results.to_csv(fullpath)

        return results


class LiRAAttack(MembershipInferenceAttack):
    def run(self):
        """Execute LiRA (Likelihood Ratio Attack)."""
        # Load target model with DP check
        self.model, target_indices = self._load_model_with_dp_check(
            self.model_dir / self.config.target_id, 
            strict=False
        )

        # Get target model confidences
        target_confs = [self.extractor.get_scaled_logits(self.model, self.device, loader) for loader in self.attack_loaders]
        target_confs = np.array(target_confs).T

        # Load intermediate results
        stats_df = self._load_intermediate_stats("scaled_logits")

        # Compute LiRA scores
        lira_scores = self._compute_lira_scores(target_confs, stats_df, self.config.offline)

        # Save results
        results = self._save_attack_results(lira_scores, target_indices, 'lira')
        return results

    def _compute_lira_scores(self, target_confs: List[float], stats_df: pd.DataFrame, offline) -> List[float]:
        """Compute LiRA scores using statistical analysis."""

        in_var, out_var, in_means, out_means = self.select_subset_shadow_metrics(stats_df)
        scores = []
        print(f"target_confs shape: {target_confs.shape}, in_means shape: {len(in_means)}, out_means shape: {len(out_means)}")

        if self.config.augment:
            print(out_var[1])

            for i, conf in enumerate(target_confs):
                if self.config.layer > 0 and i not in self.safe_indices:
                    scores.append(0.0)
                    continue
                if offline:
                    r = scipy.stats.multivariate_normal(mean=out_means[i], cov=out_var[i])
                    score = r.cdf(conf)
                else:
                    r = scipy.stats.multivariate_normal.pdf(conf, mean=out_means[i], cov=out_var[i],
                                                        allow_singular=True) + 1e-64
                    l = scipy.stats.multivariate_normal.pdf(conf, mean=in_means[i],cov=in_var[i],  allow_singular=True) + 1e-64
                    score = l / r
                scores.append(score)
        else:
            in_std = np.sqrt(in_var)
            out_std = np.sqrt(out_var)

            for i, conf in enumerate(target_confs):
                if offline:
                    z_score = (conf - out_means[i]) / out_std[i]
                    score = 1 - stats.norm.cdf(z_score)
                else:
                    l = scipy.stats.norm.pdf(conf, loc=in_means[i], scale=in_std[i]) + 1e-64
                    r = scipy.stats.norm.pdf(conf, loc=out_means[i], scale=out_std[i]) + 1e-64
                    score = l / r
                scores.append(score[0])

        return scores


class RMIAAttack(MembershipInferenceAttack):
    def run(self, gamma: float = 2.0):
        """Execute RMIA"""
        # Load target model with DP check
        self.model, target_indices = self._load_model_with_dp_check(
            self.model_dir / self.config.target_id, 
            strict=False
        )

        # Get confidences and compute ratios
        target_confs = self.extractor.get_target_logits(self.model, self.device, self.attack_loaders[0])
        stats_df = self._load_intermediate_stats("target_logits")

        # Compute RMIA scores
        rmia_scores = self._compute_rmia_scores(target_confs, target_indices, stats_df, gamma)

        # Save results
        results = self._save_attack_results(rmia_scores, target_indices, f'rmia_{gamma}_new')
        return results

    def _compute_rmia_scores(self, target_confs: List[float], target_indices: List[int],
                             stats_df: pd.DataFrame, gamma: float) -> List[float]:
        """Compute RMIA scores using relative likelihood analysis."""
        test_indices = list(set(range(self.dataset_size)) - set(target_indices))

        in_conf = [[x[0] for x in dists] for k, dists in stats_df['in_conf'].items()]
        out_conf = [[x[0] for x in dists] for k, dists in stats_df['out_conf'].items()]

        if self.config.n_shadows:
            if self.config.shadow_seed:
                np.random.seed(self.config.shadow_seed)
            selected_idx = np.random.choice(range(len(stats_df['in_conf'][0])), self.config.n_shadows, replace=False)

            out_conf = [[c[i] for i in selected_idx] for c in out_conf]
            in_conf = [[c[i] for i in selected_idx] for c in in_conf]

        print(f"n_shadows={self.config.n_shadows}, in_confs={len(in_conf[0])}, out_confs={len(out_conf[0])}")

        ratio_z = [target_confs[z] / np.mean(out_conf[z]) for z in test_indices]

        scores = []

        for i, conf in enumerate(target_confs):
            pr_x = (np.sum(in_conf[i]) + np.sum(out_conf[i])) / (2 * len(out_conf[i]))
            ratio_x = conf / pr_x
            scores.append(sum((ratio_x / r_z) > gamma for r_z in ratio_z))

        return scores

class AttackR(MembershipInferenceAttack):
    def run(self):
        """Execute RMIA (Relative Membership Inference Attack)."""
        # Load target model with DP check
        self.model, target_indices = self._load_model_with_dp_check(
            self.model_dir / self.config.target_id, 
            strict=False
        )

        # Get confidences and compute ratios
        target_confs = self.extractor.get_losses(self.model, self.device, self.attack_loaders[0])
        stats_df = self._load_intermediate_stats("losses")

        # Save results
        attackr_scores, thresholds, preds = self._compute_attackr_scores(target_confs, target_indices, stats_df, 0)
        results = self._save_attack_results(attackr_scores, target_indices, f'attackr', thresholds=thresholds, preds=preds)
        return results

    @staticmethod
    def inverse_quantile(data, value):
        """Inverse of np.quantile with linear interpolation"""
        data = np.asarray(data)
        data = np.append(data, 0)
        data = np.append(data, 1000)
        sorted_data = np.sort(data)

        # Generate uniform quantiles matching numpy's behavior
        quantiles = np.linspace(0, 1, len(data))

        # Interpolate to find the quantile for the given value
        return np.interp(value, sorted_data, quantiles)

    @staticmethod
    def calculate_loss_threshold(alpha, distribution):
        threshold = np.quantile(distribution, q=alpha, method='lower')
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

        train_thresholds = []
        train_percs = []
        preds = []

        threshold_func = self.calculate_loss_threshold if alpha < 0.001 else self.linear_itp_threshold_func

        out_conf = [[x[0] for x in dists] for k,dists in stats_df['out_conf'].items()]

        if self.config.n_shadows:
            if self.config.shadow_seed:
                np.random.seed(self.config.shadow_seed)
            selected_idx = np.random.choice(range(len(stats_df['in_conf'][0])), self.config.n_shadows, replace=False)
            out_conf = [[c[i] for i in selected_idx] for c in out_conf]

        for point_idx, point_loss_dist in enumerate(out_conf):
            threshold = threshold_func(alpha=alpha, distribution=point_loss_dist)
            perc = self.inverse_quantile(point_loss_dist, target_confs[point_idx])
            train_thresholds.append(threshold)
            train_percs.append(perc)

            preds.append(1) if target_confs[point_idx] <= threshold else preds.append(0)

        return train_percs, train_thresholds, preds



def parse_args() -> AttackConfig:
    """Parse command line arguments and return attack configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--n_shadows', type=int, default=None, nargs='?')
    parser.add_argument('--arch', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batchsize', type=int, default=500)
    parser.add_argument('--target_id', default='target', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--attack', type=str, required=True, choices=['LiRA', 'RMIA', 'AttackR'])
    parser.add_argument('--layer', type=int, default=0, help='Number of layers to use for the attack, default is 0 (all layers)')
    parser.add_argument('--layer_folder', type=str, default=None, help='Folder to retrieve layer-indices from, if applicable')
    
    args = parser.parse_args()

    # Load saved configurations if available
    model_dir = Path(MODEL_DIR) / args.exp_id
    if args.checkpoint:
        model_dir = model_dir / f'checkpoint_before_{args.checkpoint}'

    device = torch.device(f"cuda{args.gpu}" if torch.cuda.is_available() else "cpu")
    try:
        saves = torch.load(model_dir / 'shadow_0', map_location=device)
        args.arch = args.arch or saves['arch']
        args.dataset = args.dataset or saves['dataset']
        args.augment = saves['hyperparameters']['augment']
    except Exception as e:
        print(f"Warning: Could not load all settings: {e}")

    print(f"agument={args.augment}, arch={args.arch}, dataset={args.dataset}")

    return AttackConfig(
        exp_id=args.exp_id,
        target_id=args.target_id,
        checkpoint=args.checkpoint,
        arch=args.arch,
        dataset=args.dataset,
        attack=args.attack,
        n_shadows=args.n_shadows,
        offline=args.offline,
        augment=args.augment,
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        gpu=args.gpu,
        layer=args.layer,
        layer_folder=args.layer_folder if args.layer > 0 else None
    )


def main():
    config = parse_args()
    set_seed()

    # Initialize and run appropriate attack
    if config.attack == 'LiRA':
        attack = LiRAAttack(config)
    elif config.attack == "RMIA":
        attack = RMIAAttack(config)
    else: 
        attack = AttackR(config)

    attack.run()


if __name__ == "__main__":
    main()
