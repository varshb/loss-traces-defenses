#!/usr/bin/env python3
"""
Membership Inference Attack Pipeline Runner

This script serves as an entry point to run the full membership inference attack pipeline
end-to-end. The pipeline consists of three main stages with default settings:

1. Train a target model on CIFAR10 dataset using WideResNet28-2 architecture
2. Train 256 shadow models with the same architecture and dataset
3. Run membership inference attacks (LiRA, RMIA, AttackR) on the target model

The script manages experiment IDs, handles intermediate results, and provides progress tracking.
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, List

from loss_traces.config import MODEL_DIR, STORAGE_DIR


class AttackPipelineRunner:
    """Manages the full membership inference attack pipeline."""
    
    def __init__(self, exp_id: str, target: str = 'target', arch: str = "wrn28-2", dataset: str = "CIFAR10", 
                 n_shadows: int = 64, epochs: int = 100, gpu: str = "", seed: int = 2546,
                 # Differential Privacy parameters
                 private: bool = False, clip_norm: float = None, noise_multiplier: float = None,
                 target_epsilon: float = None, target_delta: float = 1e-5, layer: int = 0, layer_folder: Optional[str] = None,
                 augmult: int = 0, selective_clip: bool = False,
                 model_start: int = None, model_stop: int = None,
                 batchsize: int = None):
        """
        Initialize the pipeline runner.
        
        Args:
            exp_id: Experiment identifier for saving models and results
            arch: Model architecture (default: wrn28-2 for WideResNet28-2)
            dataset: Dataset to train on (default: CIFAR10)
            n_shadows: Number of shadow models to train (default: 256)
            gpu: GPU specification (e.g., ":0" or "")
            seed: Random seed for reproducibility
            epochs: Number of epochs to train
            private: Enable differential privacy training
            clip_norm: Clipping norm for per-sample gradients (required for DP)
            noise_multiplier: Noise multiplier for DP training
            target_epsilon: Target epsilon for DP (privacy budget)
            target_delta: Target delta for DP (privacy parameter)
            layer: Layer index for removed vulnerable points (default: 0)
        """
        self.exp_id = exp_id
        self.target = target  # Target model identifier
        self.arch = arch
        self.dataset = dataset
        self.n_shadows = n_shadows
        self.gpu = gpu
        self.seed = seed
        self.layer = layer  # Layer index for removed vulnerable points
        self.layer_folder = layer_folder
        self.model_start = model_start
        self.model_stop = model_stop

        # Training hyperparameters
        self.augmult = augmult
        self.epochs = epochs
        self.weight_decay = 5e-4
        self.momentum = 0.9
        
        # Differential Privacy parameters
        self.private = private
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.selective_clip = selective_clip
        if batchsize is None:
            self.batchsize = 1024 if self.noise_multiplier else 256
        else:
            self.batchsize = batchsize
        self.lr = 0.1

        # Paths
        self.model_dir = Path(MODEL_DIR) / self.exp_id
        self.storage_dir = Path(STORAGE_DIR)
        
        print(f"Initialized Attack Pipeline for experiment: {self.exp_id}")
        print(f"Architecture: {self.arch}, Dataset: {self.dataset}")
        print(f"Shadow models: {self.n_shadows}, GPU: {self.gpu or 'CPU'}")
        
        # Print DP settings if enabled
        if self.private:
            print(f"🔒 Differential Privacy ENABLED:")
            print(f"  - Target epsilon: {self.target_epsilon}")
            print(f"  - Target delta: {self.target_delta}")
            print(f"  - Clip norm: {self.clip_norm}")
        elif self.clip_norm or self.noise_multiplier:
            print(f"🔒 Gradient clipping/noise ENABLED:")
            if self.clip_norm:
                print(f"  - Clip norm: {self.clip_norm}")
            if self.noise_multiplier:
                print(f"  - Noise multiplier: {self.noise_multiplier}")
        
        print(f"Model directory: {self.model_dir}")
        print(f"Storage directory: {self.storage_dir}")

    def _run_command(self, cmd: list, description: str, timeout: Optional[int] = None) -> int:
        """
        Execute a command with proper logging and error handling, streaming output in real-time.
        
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
            # Start subprocess with streaming output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Stream output line by line
            output_lines = []
            while True:
                line = process.stdout.readline()
                if line:
                    # Print line immediately and store it
                    print(line.rstrip())
                    output_lines.append(line)
                    sys.stdout.flush()  # Ensure immediate display
                else:
                    # Check if process has finished
                    if process.poll() is not None:
                        break
                
                # Check for timeout
                if timeout and (time.time() - start_time) > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise subprocess.TimeoutExpired(cmd, timeout)
            
            # Wait for process to complete and get return code
            return_code = process.wait()
            elapsed_time = time.time() - start_time
            
            if return_code == 0:
                print(f"✅ SUCCESS: {description} completed in {elapsed_time:.2f}s")
            else:
                print(f"❌ FAILED: {description} failed with return code {return_code}")
                    
            return return_code
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            print(f"❌ TIMEOUT: {description} timed out after {elapsed_time:.2f}s")
            return -1
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ ERROR: {description} failed with exception after {elapsed_time:.2f}s: {e}")
            return -1

    def _check_target_model_exists(self) -> bool:
        """Check if target model already exists."""
        target_path = Path(f"{self.model_dir}/{self.target}")
        return target_path.exists()

    def _check_shadow_models_exist(self) -> int:
        """Check how many shadow models already exist."""
        count = 0
        for i in range(self.n_shadows):
            shadow_path = self.model_dir / f"shadow_{i}"
            if shadow_path.exists():
                count += 1
        return count

    def _check_attack_results_exist(self, attack_type: str, n_shadows: int) -> bool:
        """Check if attack results already exist for a specific attack type."""
        results_dir = self.storage_dir / f"{attack_type.lower()}_scores"
        result_file = results_dir / f"{self.exp_id}_target_{n_shadows}"
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
            "python3", "-m", "loss_traces.main",
            "--arch", self.arch,
            # "--track_computed_loss",  # Required for attack
            "--gpu", self.gpu,
            "--dataset", self.dataset,
            "--seed", str(self.seed),
            "--batchsize", str(self.batchsize),
            "--lr", str(self.lr),
            "--epochs", str(self.epochs),
            "--augment",  # Use data augmentation
            "--weight_decay", str(self.weight_decay),
            "--momentum", str(self.momentum),
            "--exp_id", self.exp_id,
            "--layer", str(self.layer),  # Layer index for removed vulnerable points
            "--layer_folder", str(self.layer_folder) if self.layer > 0 else "",
        ]
        if self.augmult == 0:
            cmd.extend(["--track_computed_loss"])
        if self.augmult > 0:
            cmd.extend(["--augmult", str(self.augmult)])
        if self.selective_clip:
            cmd.extend(["--selective_clip"])
        # Add differential privacy parameters if specified
        if self.private:
            cmd.extend(["--private"])
            if self.target_epsilon is not None:
                cmd.extend(["--target_epsilon", str(self.target_epsilon)])
            cmd.extend(["--target_delta", str(self.target_delta)])
            if self.clip_norm is not None:
                cmd.extend(["--clip_norm", str(self.clip_norm)])
        else:
            # Add individual DP components if specified without full DP
            if self.clip_norm is not None:
                cmd.extend(["--clip_norm", str(self.clip_norm)])
            if self.noise_multiplier is not None:
                cmd.extend(["--noise_multiplier", str(self.noise_multiplier)])
        
        return_code = self._run_command(
            cmd, 
            f"Training target model ({self.arch} on {self.dataset})" + 
            (f" with DP (ε={self.target_epsilon})" if self.private else ""),
            timeout=25200  # 2 hours timeout
        )
        
        return return_code == 0

    def train_shadow_models(self, force_retrain: bool = False) -> bool:
        """
        Train all shadow models at once.
        
        Args:
            force_retrain: If True, retrain all models even if they exist
            
        Returns:
            True if successful, False otherwise
        """
        existing_count = self._check_shadow_models_exist()
        
        if not force_retrain and existing_count == self.n_shadows:
            print(f"✅ All {self.n_shadows} shadow models already exist for {self.exp_id}")
            return True
        elif existing_count > 0:
            print(f"Found {existing_count}/{self.n_shadows} existing shadow models")

        if self.model_start is None:
            self.model_start = str(existing_count)
        if self.model_stop is None:
            self.model_stop = str(self.n_shadows)

        cmd = [
            "python3", "-m", "loss_traces.main",
            "--arch", self.arch,
            "--gpu", self.gpu,
            "--dataset", self.dataset,
            "--seed", str(self.seed),
            "--batchsize", str(self.batchsize),
            "--lr", str(self.lr),
            "--epochs", str(self.epochs),
            "--augment",  # Use data augmentation
            "--weight_decay", str(self.weight_decay),
            "--momentum", str(self.momentum),
            "--exp_id", self.exp_id,
            "--shadow_count", str(self.n_shadows),
            "--model_start", str(self.model_start),
            "--model_stop", str(self.model_stop),
            "--layer", str(self.layer),  # Layer index for removed vulnerable points
            "--layer_folder", str(self.layer_folder) if self.layer > 0 else "",
        ]
        
        if self.augmult > 0:
            cmd.extend(["--augmult", str(self.augmult)])
        if self.selective_clip:
            cmd.extend(["--selective_clip"])
        # Add differential privacy parameters if specified
        if self.private:
            cmd.extend(["--private"])
            if self.target_epsilon is not None:
                cmd.extend(["--target_epsilon", str(self.target_epsilon)])
            cmd.extend(["--target_delta", str(self.target_delta)])
            if self.clip_norm is not None:
                cmd.extend(["--clip_norm", str(self.clip_norm)])
        else:
            # Add individual DP components if specified without full DP
            if self.clip_norm is not None:
                cmd.extend(["--clip_norm", str(self.clip_norm)])
            if self.noise_multiplier is not None:
                cmd.extend(["--noise_multiplier", str(self.noise_multiplier)])
        
        return_code = self._run_command(
            cmd,
            f"Training all {self.n_shadows} shadow models" + 
            (f" with DP (ε={self.target_epsilon})" if self.private else ""),
            timeout=None
        )
        
        if return_code == 0:
            print(f"✅ Successfully trained all {self.n_shadows} shadow models")
            return True
        else:
            print(f"❌ Failed to train shadow models")
            return False

    def run_attack(self, attack_type: str, force_rerun: bool = False) -> bool:
        """
        Run a specific membership inference attack on the target model.
        
        Args:
            attack_type: Type of attack to run ("LiRA", "RMIA", or "AttackR")
            force_rerun: If True, rerun even if results exist
            
        Returns:
            True if successful, False otherwise
        """
        n_shadows = self.n_shadows//2 # Number of in/out models to use for attack

        if not force_rerun and self._check_attack_results_exist(attack_type, n_shadows):
            print(f"✅ {attack_type} attack results already exist for {self.exp_id}")
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
            "python3", "-m", "loss_traces.attacks",
            "--exp_id", self.exp_id,
            "--attack", attack_type,
            "--arch", self.arch,
            "--dataset", self.dataset,
            "--gpu", self.gpu,
            "--target_id", self.target,
            "--n_shadows", str(n_shadows), 
            "--layer", str(self.layer),  # Layer index for removed vulnerable points
            "--layer_folder", str(self.layer_folder)
        ]
        return_code = self._run_command(
            cmd,
            f"Running {attack_type} attack on {self.exp_id}",
            timeout=8000  # 1 hour timeout
        )
        
        return return_code == 0

    def run_lira_attack(self, force_rerun: bool = False) -> bool:
        """
        Run LiRA (Likelihood Ratio Attack) on the target model.
        
        Args:
            force_rerun: If True, rerun even if results exist
            
        Returns:
            True if successful, False otherwise
        """
        return self.run_attack("LiRA", force_rerun)

    def run_rmia_attack(self, force_rerun: bool = False) -> bool:
        """
        Run RMIA (Reference Model based Inference Attack) on the target model.
        
        Args:
            force_rerun: If True, rerun even if results exist
            
        Returns:
            True if successful, False otherwise
        """
        return self.run_attack("RMIA", force_rerun)

    def run_attackr_attack(self, force_rerun: bool = False) -> bool:
        """
        Run AttackR on the target model.
        
        Args:
            force_rerun: If True, rerun even if results exist
            
        Returns:
            True if successful, False otherwise
        """
        return self.run_attack("AttackR", force_rerun)

    def run_all_attacks(self, force_rerun: bool = False, 
                       attacks: List[str] = ["LiRA", "RMIA", "AttackR"]) -> bool:
        """
        Run all specified membership inference attacks on the target model.
        
        Args:
            force_rerun: If True, rerun even if results exist
            attacks: List of attacks to run
            
        Returns:
            True if all attacks successful, False otherwise
        """
        success_count = 0
        attacks = ["LiRA"]  # Default to LiRA only for now
        for attack_type in attacks:
            print(f"\n🎯 Running {attack_type} Attack")
            if self.run_attack(attack_type, force_rerun):
                success_count += 1
            else:
                print(f"❌ {attack_type} attack failed")
        
        if success_count == len(attacks):
            print(f"✅ All {len(attacks)} attacks completed successfully")
            return True
        else:
            print(f"❌ {success_count}/{len(attacks)} attacks succeeded")
            return False

    def run_full_pipeline(self, force_retrain: bool = False, 
                         attacks: List[str] = ["LiRA", "RMIA", "AttackR"]) -> bool:
        """
        Run the complete attack pipeline end-to-end.
        
        Args:
            force_retrain: If True, retrain models even if they exist
            attacks: List of attacks to run (default: all available attacks)
            
        Returns:
            True if entire pipeline succeeds, False otherwise
        """
        print(f"\n🚀 STARTING FULL MEMBERSHIP INFERENCE ATTACK PIPELINE")
        print(f"Experiment ID: {self.exp_id}")
        print(f"Force retrain: {force_retrain}")
        print(f"Attacks to run: {', '.join(attacks)}")
        
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
            
        # Stage 3: Run attacks
        print(f"\n🎯 STAGE 3/3: Running Membership Inference Attacks")
        if not self.run_all_attacks(force_retrain, attacks):
            print("❌ Pipeline failed during attack execution")
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
        attacks = ["LiRA", "RMIA", "AttackR"]
        print(f"\nAttack Results:")
        for attack in attacks:
            results_exist = self._check_attack_results_exist(attack)
            print(f"  {attack}: {'✅ EXISTS' if results_exist else '❌ MISSING'}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if not target_exists:
            print("   • Run target model training first")
        elif shadow_count < self.n_shadows:
            print(f"   • Train remaining {self.n_shadows - shadow_count} shadow models")
        else:
            missing_attacks = [attack for attack in attacks 
                             if not self._check_attack_results_exist(attack)]
            if missing_attacks:
                print(f"   • Run missing attacks: {', '.join(missing_attacks)}")
            else:
                print("   • All components ready! Pipeline has been completed.")


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
  # Run full pipeline with default settings (all attacks)
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo
  
  # Run with custom settings
  python run_attack_pipeline.py --exp_id my_experiment --n_shadows 32 --gpu :0
  
  # Run with differential privacy (full DP)
  python run_attack_pipeline.py --exp_id dp_experiment --private --target_epsilon 8.0 --clip_norm 1.0
  
  # Run with gradient clipping only (no noise)
  python run_attack_pipeline.py --exp_id clipped_experiment --clip_norm 1.0
  
  # Run with custom DP parameters
  python run_attack_pipeline.py --exp_id custom_dp --clip_norm 0.5 --noise_multiplier 1.1
  
  # Check status only
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --status
  
  # Force retrain everything
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --full --force
  
  # Run individual stages
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --target-only
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --shadows-only
  
  # Run specific attacks
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --lira-only
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --rmia-only
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --attackr-only
  python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --attacks-only
        """
    )
    
    # Required arguments
    parser.add_argument("--exp_id", type=str, required=True,
                      help="Experiment identifier for saving models and results")
    parser.add_argument("--target", type=str, default="target",
                      help="Target model identifier (default: 'target')")
    
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
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of epochs to train (default: 100)")                  
    parser.add_argument("--layer", type=int, default=0,
                      help="Layer index for removed vulnerable points (default: 0)")
    parser.add_argument("--layer_folder", type=str, default=None,
                      help="Folder to retrieve layer-indices from, if applicable")
    parser.add_argument("--model_start", type=int, default=None,
                      help="Starting model index for shadow training (default: None)")
    parser.add_argument("--model_stop", type=int, default=None,
                        help="Stopping model index for shadow training (default: None)")
    parser.add_argument("--batchsize", type=int, default=None,
                      help="Batch size for training (default: 256)")
    # Differential Privacy arguments
    parser.add_argument("--private", action="store_true",
                      help="Enable differential privacy training")
    parser.add_argument("--clip_norm", type=float, default=None,
                      help="Clipping norm for per-sample gradients (required for DP)")
    parser.add_argument("--noise_multiplier", type=float, default=None,
                      help="Noise multiplier for DP training")
    parser.add_argument("--target_epsilon", type=float, default=None,
                      help="Target epsilon for DP (privacy budget)")
    parser.add_argument("--target_delta", type=float, default=1e-5,
                      help="Target delta for DP (default: 1e-5)")
    parser.add_argument("--augmult", type=int, default=0,
                      help="Enable data augmentation multiplicatively")
    parser.add_argument("--selective_clip", action="store_true",
                      help="Enable selective clipping")
    # Pipeline control
    parser.add_argument("--full", action="store_true",
                      help="Run the complete pipeline end-to-end (all attacks)")
    parser.add_argument("--target-only", action="store_true",
                      help="Train target model only")
    parser.add_argument("--shadows-only", action="store_true",
                      help="Train shadow models only")
    parser.add_argument("--attacks-only", action="store_true",
                      help="Run all attacks only")
    parser.add_argument("--lira-only", action="store_true",
                      help="Run LiRA attack only")
    parser.add_argument("--rmia-only", action="store_true",
                      help="Run RMIA attack only")
    parser.add_argument("--attackr-only", action="store_true",
                      help="Run AttackR attack only")
    parser.add_argument("--status", action="store_true",
                      help="Show current status of the experiment")
    parser.add_argument("--force", action="store_true",
                      help="Force retrain/rerun even if outputs exist")
    
    # Attack selection
    parser.add_argument("--attack-types", nargs="+", 
                      choices=["LiRA", "RMIA", "AttackR"],
                      default=["LiRA", "RMIA", "AttackR"],
                      help="Specify which attacks to run (default: all)")
    
    args = parser.parse_args()
    
    # Create pipeline runner
    runner = AttackPipelineRunner(
        exp_id=args.exp_id,
        target=args.target,
        arch=args.arch,
        dataset=args.dataset,
        n_shadows=args.n_shadows,
        gpu=args.gpu,
        seed=args.seed,
        epochs=args.epochs,
        # Differential Privacy parameters
        private=args.private,
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        layer=args.layer,
        layer_folder=args.layer_folder if args.layer > 0 else None,
        augmult=args.augmult,
        selective_clip=args.selective_clip,
        model_start=args.model_start,
        model_stop=args.model_stop,
        batchsize=args.batchsize
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
    elif args.attacks_only:
        success = runner.run_all_attacks(args.force, args.attack_types)
        sys.exit(0 if success else 1)
    elif args.lira_only:
        success = runner.run_lira_attack(args.force)
        sys.exit(0 if success else 1)
    elif args.rmia_only:
        success = runner.run_rmia_attack(args.force)
        sys.exit(0 if success else 1)
    elif args.attackr_only:
        success = runner.run_attackr_attack(args.force)
        sys.exit(0 if success else 1) 
    elif args.full:
        success = runner.run_full_pipeline(args.force, args.attack_types)
        sys.exit(0 if success else 1)
    else:
        # Default: show status and offer to run full pipeline
        runner.status()
        print(f"\n💡 To run the full pipeline, use: --full")
        print(f"💡 To see all options, use: --help")


if __name__ == "__main__":
    main() 