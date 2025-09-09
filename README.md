# Code for "The Tallest Trees Catch the Most Wind: A Cost-Effective Pruning Strategy for High-Utility Membership Inference Defense"

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

Most of the results presented in the paper use the setup with WideResNet28-2 trained on CIFAR-10, with 64 shadow models.

Some additional experiments, however, use other and datasets (CINIC-10).

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

# Running Experiments

## 1. Run the Complete Pipeline

The `scripts/` folder contains experiment scripts for different defense strategies and evaluations.

### Iterative Removal
This pipeline will:
- Train a **WideResNet28-2** target model on CIFAR-10 (100 epochs) with **loss trajectory tracking**  
- Train **64 shadow models** with the same architecture  
- Run the **LiRA membership inference attacks**  
- Compute **LT-IQR vulnerability scores** for all training samples  
- Iteratively remove the top *k%* most vulnerable samples and retrain the model  
- Evaluate **privacy leakage** at each removal layer using LiRA  

### Selective Clipping
This pipeline will:
- Train a **WideResNet28-2** target model on CIFAR-10  
- Train **64 shadow models** with **selective gradient clipping** applied to vulnerable samples  
- Compute **LT-IQR vulnerability scores** to identify samples requiring protection  
- Apply **gradient clipping** only to the top *k%* most vulnerable samples during training  
- Run **LiRA attacks** to evaluate privacy protection effectiveness  


> ⚠️ **Disclaimer:** Selective clipping experiments **cannot be run without first training Layer 0 of the iterative removal experiment**, since they rely on the vulnerability scores produced in that step.

---

## 2. Individual Experiment Scripts

The `scripts/` folder contains the following:

- **`cifar_top2_removal_experiments.sh`** — Iterative removal of top 2% most vulnerable samples  
- **`cifar_top5_removal_experiments.sh`** — Iterative removal of top 5% most vulnerable samples  
- **`random_removal_experiments.sh`** — Removal of randomly selected samples for comparison  
- **`selective_clipping_experiments.sh`** — Run selective gradient clipping experiments (requires Layer 0 from iterative removal)  
- **`standard_clipping_experiments.sh`** — Run standard (uniform) gradient clipping experiments  
- **`standard_dp_sgd.sh`** — Run standard DP-SGD experiments  
- **`enhanced_dp_sgd.sh`** — Run enhanced DP-SGD experiments with AugMult factor 4


The training is performed on a single GPU and is rather time-consuming. Full training run with 64 shadow models
is expected to take from ~12 hours to 78 hours for utility-enhanced DP-SGD.


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
└── shadow_63                 # Shadow model 64

{storage_dir}/losses/
└── {exp_id}_target          # Computed loss traces

{storage_dir}/lira_scores/
└── {exp_id}_target_{n_shadows}          # LiRA attack results (CSV)
```

## Analysis

We provide the notebooks reproducing the core claims and figures from the paper in `notebooks/`.

* `iter_removal.ipynb`: Main results for both iterative removal and selective clipping.
* `layer_overlap.ipynb`: Additional plots that compare 
