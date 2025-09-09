arch="wrn28-2"
dataset="CIFAR10"
clip_norm=10

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for noise in 0.01 0.1 0.2; do
    echo "Running DP-SGD with noise ${noise} and clip norm ${clip_norm}"
    python -m src.loss_traces.run_attack_pipeline \
        --exp_id dp_${noise}n_${clip_norm}c \
        --arch ${arch}  \
        --dataset ${dataset} \
        --layer 0 \
        --target-only \
        --clip_norm ${clip_norm} \
        --noise_multiplier ${noise}

    python -m src.loss_traces.run_attack_pipeline \
        --exp_id dp_${noise}n_${clip_norm}c \
        --arch ${arch} \
        --dataset ${dataset} \
        --layer 0 \
        --shadows-only \
        --clip_norm ${clip_norm} \
        --noise_multiplier ${noise} 

    python -m src.loss_traces.run_attack_pipeline \
        --exp_id dp_${noise}n_${clip_norm}c \
        --arch ${arch} \
        --dataset ${dataset} \
        --layer 0 \
        --lira-only \
        --clip_norm ${clip_norm} \
        --noise_multiplier ${noise} 
done
