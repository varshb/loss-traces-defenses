# Membership Inference Attack Pipeline

This repository provides a complete, reproducible pipeline for conducting membership inference attacks using the LiRA (Likelihood Ratio Attack) method. The pipeline trains a target model, multiple shadow models, and then performs the attack.

## Setup

To install dependencies, run:
```

pip install -r requirements.txt

```
Next create a config.py file with the following and fill in the relevant paths:

```

LOCAL_DIR = # path to this folder
# paths to store stuff...
STORAGE_DIR = 
MY_STORAGE_DIR = 
MODEL_DIR = 
DATA_DIR = 

```

## Quick Start

### 1. Run the Complete Pipeline

```bash
cd loss_traces
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --full
```

This will:
1. Train a WideResNet28-2 target model on CIFAR10 (100 epochs)
2. Train 256 shadow models with the same architecture  
3. Run the LiRA membership inference attack

### 2. Check Status

```bash
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --status
```

### 3. Run Individual Stages

```bash
# Train only the target model
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --target-only

# Train only the shadow models
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --shadows-only

# Run only the attack (requires target and shadow models to exist)
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --attack-only
```

## Configuration Options

### Basic Options

```bash
python run_attack_pipeline.py \
    --exp_id my_experiment \
    --arch wrn28-2 \
    --dataset CIFAR10 \
    --n_shadows 256 \
    --gpu :0 \
    --seed 2546 \
    --full
```

### Advanced Options

- `--force`: Force retrain/rerun even if outputs already exist
- `--target-only`: Train target model only
- `--shadows-only`: Train shadow models only  
- `--attack-only`: Run attack only
- `--status`: Show current experiment status

## Architecture Support

The pipeline supports the following architectures (via `--arch`):

- `wrn28-2`: WideResNet28-2 (default)
- `wrn28-10`: WideResNet28-10
- `wrn40-4`: WideResNet40-4


## Dataset Support

Currently supported datasets (via `--dataset`):
- `CIFAR10` (default)
- `CIFAR100`
- `CINIC10`

## Training Configuration

The pipeline uses optimized hyperparameters for WideResNet28-2 on CIFAR10:

- **Batch size**: 256
- **Learning rate**: 0.1  
- **Epochs**: 100
- **Weight decay**: 5e-4
- **Momentum**: 0.9
- **Data augmentation**: Enabled
- **Optimizer**: SGD with Cosine Annealing

## Output Structure

The pipeline creates the following directory structure:

```
{storage_dir}/trained_models/{exp_id}/
├── target                    # Target model
├── shadow_0                  # Shadow model 0
├── shadow_1                  # Shadow model 1
├── ...
└── shadow_256                 # Shadow model 256

{storage_dir}/lira_scores/
└── {exp_id}_target          # LiRA attack results (CSV)

{storage_dir}/scaled_logits_intermediate/
└── {exp_id}.pt              # Intermediate statistics for attack
```

## Examples

### Example 1: Quick Demo with Fewer Shadow Models

```bash
python run_attack_pipeline.py \
    --exp_id quick_demo \
    --n_shadows 16 \
    --full
```

### Example 2: Different Architecture

```bash
python run_attack_pipeline.py \
    --exp_id resnet_experiment \
    --arch rn-18 \
    --n_shadows 32 \
    --full
```

### Example 3: GPU Training

```bash
python run_attack_pipeline.py \
    --exp_id gpu_experiment \
    --gpu :0 \
    --full
```

### Example 4: Resume Interrupted Training

```bash
# Check what's already completed
python run_attack_pipeline.py --exp_id my_experiment --status

# Continue from where it left off
python run_attack_pipeline.py --exp_id my_experiment --full
```

### Example 5: Force Complete Retrain

```bash
python run_attack_pipeline.py \
    --exp_id my_experiment \
    --full \
    --force
```

## Expected Runtime

Approximate times on a modern GPU:

- **Target model training**: 2-4 hours (200 epochs)
- **64 Shadow models**: 8-16 hours (training in parallel chunks)
- **LiRA attack**: 30-60 minutes
- **Total pipeline**: 10-20 hours

Times scale roughly linearly with the number of shadow models.

## Results Interpretation

After completion, the attack results are saved as a CSV file containing:

- `lira_score`: The LiRA attack score for each sample
- `target_trained_on`: Boolean indicating if sample was in target training set
- `og_idx`: Original dataset index

Higher LiRA scores indicate higher likelihood of membership in the training set.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU training
2. **Disk space**: The pipeline requires ~10-50GB depending on model size
3. **Interrupted training**: Use `--status` to check progress and resume

### Resume After Interruption

The pipeline automatically skips completed components:

```bash
# This will skip any existing models and continue from where it left off
python run_attack_pipeline.py --exp_id my_experiment --full
```

### Clean Restart

```bash
# Force complete retrain
python run_attack_pipeline.py --exp_id my_experiment --full --force
```

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Required packages (see `requirements.txt`)
- Properly configured `config.py` with data paths

## Files Generated

The pipeline generates several types of files:

1. **Model files**: Trained PyTorch models with weights and metadata
2. **Attack results**: CSV files with membership inference scores  
3. **Intermediate statistics**: Cached statistics for faster attack reruns
4. **Logs**: Training progress and attack execution logs



## References

[^1]: N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramer,
“Membership inference attacks from first principles,” in 2022 IEEE
Symposium on Security and Privacy (SP). IEEE, 2022, pp. 1897–
1914

[^2]: J. Ye, A. Maddi, S. K. Murakonda, V. Bindschaedler, and R. Shokri,
“Enhanced membership inference attacks against machine learning
models,” in Proceedings of the 2022 ACM SIGSAC Conference on
Computer and Communications Security, 2022, pp. 3093–3106.
