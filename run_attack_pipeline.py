#!/usr/bin/env python3
"""
Membership Inference Attack Pipeline Runner

This script serves as an entry point to run the full membership inference attack pipeline
end-to-end, ensuring reproducibility. The pipeline consists of three main stages:

1. Train a target model on CIFAR10 dataset using WideResNet28-2 architecture
2. Train 64 shadow models with the same architecture and dataset
3. Run LiRA (Likelihood Ratio Attack) on the target model

The script manages experiment IDs, handles intermediate results, and provides progress tracking.
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_DIR, STORAGE_DIR


class AttackPipelineRunner:
    """Manages the full membership inference attack pipeline."""
    
    def __init__(self, exp_id: str, arch: str = "wrn28-2", dataset: str = "CIFAR10", 
                 n_shadows: int = 64, gpu: str = "", seed: int = 2546):
        """
        Initialize the pipeline runner.
        
        Args:
            exp_id: Experiment identifier for saving models and results
            arch: Model architecture (default: wrn28-2 for WideResNet28-2)
            dataset: Dataset to train on (default: CIFAR10)
            n_shadows: Number of shadow models to train (default: 64)
            gpu: GPU specification (e.g., ":0" or "")
            seed: Random seed for reproducibility
        """
        self.exp_id = exp_id
        self.arch = arch
        self.dataset = dataset
        self.n_shadows = n_shadows
        self.gpu = gpu
        self.seed = seed
        
        # Training hyperparameters optimized for WideResNet28-2 on CIFAR10
        self.batchsize = 256  # As per HPC config for wrn28-2
        self.lr = 0.1
        self.epochs = 200
        self.weight_decay = 5e-4
        self.momentum = 0.9
        
        # Paths
        self.model_dir = Path(MODEL_DIR) / self.exp_id
        self.storage_dir = Path(STORAGE_DIR)
        
        print(f"Initialized Attack Pipeline for experiment: {self.exp_id}")
        print(f"Architecture: {self.arch}, Dataset: {self.dataset}")
        print(f"Shadow models: {self.n_shadows}, GPU: {self.gpu or 'CPU'}")
        print(f"Model directory: {self.model_dir}")
        print(f"Storage directory: {self.storage_dir}")

    def _run_command(self, cmd: list, description: str, timeout: Optional[int] = None) -> int:
        """
        Execute a command with proper logging and error handling.
        
        Args:
            cmd: Command and arguments as a list
            description: Description of the command for logging
            timeout: Optional timeout in seconds
            
        Returns:
            Return code of the command
        """
        print(f"\n{'='*60}")
        print(f"STARTING: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd, 
                check=False, 
                capture_output=True, 
                text=True,
                timeout=timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: {description} completed in {elapsed_time:.2f}s")
                if result.stdout:
                    print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            else:
                print(f"❌ FAILED: {description} failed with return code {result.returncode}")
                print("STDERR:", result.stderr)
                if result.stdout:
                    print("STDOUT:", result.stdout)
                    
            return result.returncode
            
        except subprocess.TimeoutExpired:
            print(f"❌ TIMEOUT: {description} timed out after {timeout}s")
            return -1
        except Exception as e:
            print(f"❌ ERROR: {description} failed with exception: {e}")
            return -1

    def _check_target_model_exists(self) -> bool:
        """Check if target model already exists."""
        target_path = self.model_dir / "target"
        return target_path.exists()

    def _check_shadow_models_exist(self) -> int:
        """Check how many shadow models already exist."""
        count = 0
        for i in range(self.n_shadows):
            shadow_path = self.model_dir / f"shadow_{i}"
            if shadow_path.exists():
                count += 1
        return count

    def _check_attack_results_exist(self) -> bool:
        """Check if LiRA attack results already exist."""
        results_dir = self.storage_dir / "lira_scores"
        result_file = results_dir / f"{self.exp_id}_target"
        return result_file.exists()

    def train_target_model(self, force_retrain: bool = False) -> bool:
        """
        Train the target model.
        
        Args:
            force_retrain: If True, retrain even if model exists
            
        Returns:
            True if successful, False otherwise
        """
        if not force_retrain and self._check_target_model_exists():
            print(f"✅ Target model already exists for {self.exp_id}")
            return True
            
        cmd = [
            "python3", "../main.py",
            "--arch", self.arch,
            "--track_computed_loss",  # Required for attack
            "--gpu", self.gpu,
            "--dataset", self.dataset,
            "--seed", str(self.seed),
            "--batchsize", str(self.batchsize),
            "--lr", str(self.lr),
            "--epochs", str(self.epochs),
            "--checkpoint",  # Save checkpoints
            "--augment",  # Use data augmentation
            "--weight_decay", str(self.weight_decay),
            "--momentum", str(self.momentum),
            "--exp_id", self.exp_id
        ]
        
        return_code = self._run_command(
            cmd, 
            f"Training target model ({self.arch} on {self.dataset})",
            timeout=7200  # 2 hours timeout
        )
        
        return return_code == 0

    def train_shadow_models(self, force_retrain: bool = False, 
                          chunk_size: int = 8) -> bool:
        """
        Train shadow models in chunks for efficiency.
        
        Args:
            force_retrain: If True, retrain all models even if they exist
            chunk_size: Number of models to train in each batch
            
        Returns:
            True if successful, False otherwise
        """
        existing_count = self._check_shadow_models_exist()
        
        if not force_retrain and existing_count == self.n_shadows:
            print(f"✅ All {self.n_shadows} shadow models already exist for {self.exp_id}")
            return True
        elif existing_count > 0:
            print(f"Found {existing_count}/{self.n_shadows} existing shadow models")
            
        # Train in chunks for better resource management
        success_count = 0
        for start_idx in range(0, self.n_shadows, chunk_size):
            end_idx = min(start_idx + chunk_size, self.n_shadows)
            
            # Skip if all models in this chunk already exist
            if not force_retrain:
                chunk_exists = all(
                    (self.model_dir / f"shadow_{i}").exists() 
                    for i in range(start_idx, end_idx)
                )
                if chunk_exists:
                    print(f"✅ Shadow models {start_idx}-{end_idx-1} already exist")
                    success_count += (end_idx - start_idx)
                    continue
            
            cmd = [
                "python3", "../main.py",
                "--arch", self.arch,
                "--track_computed_loss",  # Required for attack
                "--gpu", self.gpu,
                "--dataset", self.dataset,
                "--seed", str(self.seed),
                "--batchsize", str(self.batchsize),
                "--lr", str(self.lr),
                "--epochs", str(self.epochs),
                "--checkpoint",  # Save checkpoints
                "--augment",  # Use data augmentation
                "--weight_decay", str(self.weight_decay),
                "--momentum", str(self.momentum),
                "--exp_id", self.exp_id,
                "--shadow_count", str(self.n_shadows),
                "--model_start", str(start_idx),
                "--model_stop", str(end_idx)
            ]
            
            return_code = self._run_command(
                cmd,
                f"Training shadow models {start_idx}-{end_idx-1} "
                f"({end_idx-start_idx} models)",
                timeout=14400  # 4 hours timeout
            )
            
            if return_code == 0:
                success_count += (end_idx - start_idx)
            else:
                print(f"❌ Failed to train shadow models {start_idx}-{end_idx-1}")
                return False
                
        print(f"✅ Successfully trained {success_count}/{self.n_shadows} shadow models")
        return success_count == self.n_shadows

    def run_lira_attack(self, force_rerun: bool = False) -> bool:
        """
        Run LiRA (Likelihood Ratio Attack) on the target model.
        
        Args:
            force_rerun: If True, rerun even if results exist
            
        Returns:
            True if successful, False otherwise
        """
        if not force_rerun and self._check_attack_results_exist():
            print(f"✅ LiRA attack results already exist for {self.exp_id}")
            return True
            
        # Verify prerequisites exist
        if not self._check_target_model_exists():
            print("❌ Target model not found. Please train target model first.")
            return False
            
        shadow_count = self._check_shadow_models_exist()
        if shadow_count < self.n_shadows:
            print(f"❌ Only {shadow_count}/{self.n_shadows} shadow models found. "
                  f"Please train all shadow models first.")
            return False
            
        cmd = [
            "python3", "../attacks.py",
            "--exp_id", self.exp_id,
            "--attack", "LiRA",
            "--arch", self.arch,
            "--dataset", self.dataset,
            "--gpu", self.gpu,
            "--target_id", "target"
        ]
        
        return_code = self._run_command(
            cmd,
            f"Running LiRA attack on {self.exp_id}",
            timeout=3600  # 1 hour timeout
        )
        
        return return_code == 0

    def run_full_pipeline(self, force_retrain: bool = False) -> bool:
        """
        Run the complete attack pipeline end-to-end.
        
        Args:
            force_retrain: If True, retrain models even if they exist
            
        Returns:
            True if entire pipeline succeeds, False otherwise
        """
        print(f"\n🚀 STARTING FULL MEMBERSHIP INFERENCE ATTACK PIPELINE")
        print(f"Experiment ID: {self.exp_id}")
        print(f"Force retrain: {force_retrain}")
        
        pipeline_start = time.time()
        
        # Stage 1: Train target model
        print(f"\n📊 STAGE 1/3: Training Target Model")
        if not self.train_target_model(force_retrain):
            print("❌ Pipeline failed at target model training")
            return False
            
        # Stage 2: Train shadow models  
        print(f"\n👥 STAGE 2/3: Training Shadow Models ({self.n_shadows} models)")
        if not self.train_shadow_models(force_retrain):
            print("❌ Pipeline failed at shadow model training")
            return False
            
        # Stage 3: Run attack
        print(f"\n🎯 STAGE 3/3: Running LiRA Attack")
        if not self.run_lira_attack(force_retrain):
            print("❌ Pipeline failed at LiRA attack")
            return False
            
        pipeline_time = time.time() - pipeline_start
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total time: {pipeline_time:.2f}s ({pipeline_time/60:.1f} minutes)")
        print(f"Results saved in: {self.storage_dir}")
        
        return True

    def status(self):
        """Print the current status of the experiment."""
        print(f"\n📋 STATUS for experiment: {self.exp_id}")
        print(f"{'='*50}")
        
        # Target model status
        target_exists = self._check_target_model_exists()
        print(f"Target model: {'✅ EXISTS' if target_exists else '❌ MISSING'}")
        
        # Shadow models status
        shadow_count = self._check_shadow_models_exist()
        print(f"Shadow models: {shadow_count}/{self.n_shadows} "
              f"({'✅ COMPLETE' if shadow_count == self.n_shadows else '⚠️  INCOMPLETE'})")
        
        # Attack results status
        results_exist = self._check_attack_results_exist()
        print(f"LiRA results: {'✅ EXISTS' if results_exist else '❌ MISSING'}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if not target_exists:
            print("   • Run target model training first")
        elif shadow_count < self.n_shadows:
            print(f"   • Train remaining {self.n_shadows - shadow_count} shadow models")
        elif not results_exist:
            print("   • Run LiRA attack")
        else:
            print("   • All components ready! Pipeline can be executed.")


def create_experiment_id(arch: str, dataset: str, suffix: str = None) -> str:
    """Create a standardized experiment ID."""
    exp_id = f"{arch}_{dataset}"
    if suffix:
        exp_id += f"_{suffix}"
    return exp_id


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Membership Inference Attack Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo
  
  # Run with custom settings
  python run_attack_pipeline.py --exp_id my_experiment --n_shadows 32 --gpu :0
  
  # Check status only
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --status
  
  # Force retrain everything
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --full --force
  
  # Run individual stages
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --target-only
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --shadows-only
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --attack-only
        """
    )
    
    # Required arguments
    parser.add_argument("--exp_id", type=str, required=True,
                      help="Experiment identifier for saving models and results")
    
    # Model configuration
    parser.add_argument("--arch", type=str, default="wrn28-2",
                      help="Model architecture (default: wrn28-2)")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                      help="Dataset to train on (default: CIFAR10)")
    parser.add_argument("--n_shadows", type=int, default=64,
                      help="Number of shadow models to train (default: 64)")
    parser.add_argument("--gpu", type=str, default="",
                      help="GPU specification (e.g., ':0' or '')")
    parser.add_argument("--seed", type=int, default=2546,
                      help="Random seed for reproducibility (default: 2546)")
    
    # Pipeline control
    parser.add_argument("--full", action="store_true",
                      help="Run the complete pipeline end-to-end")
    parser.add_argument("--target-only", action="store_true",
                      help="Train target model only")
    parser.add_argument("--shadows-only", action="store_true",
                      help="Train shadow models only")
    parser.add_argument("--attack-only", action="store_true",
                      help="Run LiRA attack only")
    parser.add_argument("--status", action="store_true",
                      help="Show current status of the experiment")
    parser.add_argument("--force", action="store_true",
                      help="Force retrain/rerun even if outputs exist")
    
    args = parser.parse_args()
    
    # Create pipeline runner
    runner = AttackPipelineRunner(
        exp_id=args.exp_id,
        arch=args.arch,
        dataset=args.dataset,
        n_shadows=args.n_shadows,
        gpu=args.gpu,
        seed=args.seed
    )
    
    # Execute requested operation
    if args.status:
        runner.status()
    elif args.target_only:
        success = runner.train_target_model(args.force)
        sys.exit(0 if success else 1)
    elif args.shadows_only:
        success = runner.train_shadow_models(args.force)
        sys.exit(0 if success else 1)
    elif args.attack_only:
        success = runner.run_lira_attack(args.force)
        sys.exit(0 if success else 1)
    elif args.full:
        success = runner.run_full_pipeline(args.force)
        sys.exit(0 if success else 1)
    else:
        # Default: show status and offer to run full pipeline
        runner.status()
        print(f"\n💡 To run the full pipeline, use: --full")
        print(f"💡 To see all options, use: --help")


if __name__ == "__main__":
    main() 