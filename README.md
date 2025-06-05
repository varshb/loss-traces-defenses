# Code for the paper "Free Record-Level Privacy Risk Evaluation through Artifact-Based Methods"
## Abstract
Membership inference attacks (MIAs) are widely used to empirically assess privacy risks in machine learning models, both
providing model-level vulnerability metrics and identifying the most vulnerable training samples. State-of-the-art 
methods, however, require training hundreds of shadow models with the same architecture as the target model. This makes
the computational cost of assessing the privacy of models prohibitive for many practical applications, particularly when
used iteratively as part of the model development process and for large models. We propose a novel approach for identifying
the training samples most vulnerable to membership inference attacks by analyzing artifacts naturally available during
the training process. Our method, Loss Trace Interquartile Range (LT-IQR), analyzes per-sample loss trajectories 
collected during model training to identify high-risk samples without requiring any additional model training. Through
experiments on standard benchmarks, we demonstrate that LT-IQR achieves 92% precision@k= 1% in identifying the
samples most vulnerable to state-of-the-art MIAs. This result holds across datasets and model architectures with LT-IQR
outperforming both traditional vulnerability metrics, such as loss, and lightweight MIAs using few shadow models. We
also show LT-IQR to accurately identify points vulnerable to multiple MIA methods and perform ablation studies. 
We believe LT-IQR enables model developers to identify vulnerable training samples, for free, as part of the model development
process. Our results emphasize the potential of artifact-based methods to efficiently evaluate privacy risks.

## Getting started

To install the package, run:
```bash
cd loss_traces/
pip install -e .
```

This installs both the dependencies and our code.

The repo contains the code for the two stages of the analysis:

1. Shadow model training
2. Analyzing the data

Most of the results presented in the paper use the setup with WideResNet28-2 trained on CIFAR-10, with 256 shadow models.

Some additional experiments, however, use other models (WideResNet40-4, ResNet-20) and datasets (CIFAR-100, CINIC-10).

### Configuration

The project uses a configuration system with sensible defaults. The default configuration is provided in `src/loss_traces/config.py`:

```python
# Default values in src/loss_traces/config.py
STORAGE_DIR = "./data"          # For storing trained models and MIA results
DATA_DIR = "./data/datasets"    # For downloading training datasets
CINIC_10_PATH = "./cinic10"     # For pre-downloaded CINIC-10 dataset
MODEL_DIR = os.path.join(STORAGE_DIR, "models")  # Derived from STORAGE_DIR
```

To override these defaults for your local setup, create a file `src/loss_traces/config_local.py`:

```python
# Example src/loss_traces/config_local.py
STORAGE_DIR = "/path/to/your/storage"
DATA_DIR = "/path/to/your/datasets"
CINIC_10_PATH = "/path/to/cinic10"
```

The configuration system will automatically use your local overrides if the file exists.

## Training shadow models

### 1. Run the Complete Pipeline

```bash
cd loss_traces
python -m loss_traces.run_attack_pipeline \
    --exp_id wrn28-2_CIFAR10 \
    --arch wrn28-2 \
    --dataset CIFAR10 \
    --n-shadows 256 \
    --full \
    --gpu :0
```

This will:
1. Train a WideResNet28-2 target model on CIFAR10 (100 epochs)
2. Train 256 shadow models with the same architecture  
3. Run the membership inference attacks: LiRA, AttackR, RMIA

Note that `--exp_id` could be any string, but all notebooks in the analysis section expect the experiment names
to follow `{arch}_{dataset}` notation.

Run `python -m loss_traces.run_attack_pipeline --help` for additional arguments.

The training is performed on a single GPU and is rather time-consuming. Full training run with 256 shadow models
is expected to take ~72 hours.

### Architecture Support

The pipeline supports the following architectures (via `--arch`):

- `wrn28-2`: WideResNet28-2 (default)
- `rn-20`: ResNet-20
- `wrn40-4`: WideResNet40-4

### Dataset Support

Currently supported datasets (via `--dataset`):
- `CIFAR10` (default)
- `CIFAR100`
- `CINIC10`

### Training Configuration

We used the following hyperparameters to train models:

- **Batch size**: 256
- **Learning rate**: 0.1  
- **Epochs**: 100
- **Weight decay**: 5e-4
- **Momentum**: 0.9
- **Data augmentation**: Enabled

### Output Structure

The pipeline creates the following directory structure:

```
{storage_dir}/models/{exp_id}/
├── target                    # Target model
├── shadow_0                  # Shadow model 0
├── shadow_1                  # Shadow model 1
├── ...
└── shadow_256                 # Shadow model 256

{storage_dir}/losses/
└── {exp_id}_target          # Computed loss traces

{storage_dir}/lira_scores/
└── {exp_id}_target_{n_shadows}          # LiRA attack results (CSV)

{storage_dir}/attackr_scores/
└── {exp_id}_target_{n_shadows}          # AttackR attack results (CSV)

{storage_dir}/rmia_2.0_new_scores/
└── {exp_id}_target_{n_shadows}          # RMIA attack results (CSV)


{storage_dir}/{metric}_intermediate/ # Intermediate statistics for attacks
```

## Analysis

We provide the notebooks reproducing the core claims and figures from the paper in `notebooks/`.

* `main_results.ipynb`: Tables 1, 5. Figures 1, 7, 8, 11
* `additional_setups.ipynb`. Tables: 2, 3, 4, 7, 9, 10, 11, 14. 
This contains all results based non additional architectures or datasets. To fully reproduce
the results in this notebook you'll need to train shadow models for the following 5 setups:
    * WRN28-2, CIFAR-10
    * WRN28-2, CIFAR-100
    * WRN28-2, CINIC-10
    * WRN40-4, CIFAR-10
    * RN-20, CIFAR-10
* `mias.ipynb`. Table 6, Figures 3,4. Everything to do with raw MIA performance.
* `dp.ipynb`, 'stability_*.ipynb' - additional appendix results.

NB: All notebooks in the analysis section expect the experiment names to follow `{arch}_{dataset}` notation.