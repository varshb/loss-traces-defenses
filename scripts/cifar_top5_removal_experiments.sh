arch="wrn28-2"
dataset="CIFAR10"


for layer in 0 1 2 3 4 5 6; do
    echo "Running layer $layer"
    python -m src.loss_traces.run_attack_pipeline \
        --exp_id CIFAR_top5_l${layer} \
        --target target \
        --arch $arch \
        --dataset $dataset \
        --target-only \
        --layer $layer \
        --layer_folder CIFAR_top5
        
    python src/privacy_onion/layer_utils.py \
        --exp_id CIFAR_top5_l${layer} \
        --layer $layer \
        --exp_path CIFAR_top5 \
        --top_k 0.05

    echo "Training shadows for layer $layer"
    python -m src.loss_traces.run_attack_pipeline \
        --exp_id CIFAR_top5_l${layer} \
        --target target \
        --arch $arch \
        --dataset $dataset \
        --shadows-only \
        --layer $layer \
        --layer_folder CIFAR_top5
    
    python -m src.loss_traces.run_attack_pipeline \
        --exp_id CIFAR_top5_l${layer} \
        --target target \
        --arch $arch \
        --dataset $dataset \
        --lira-only \
        --layer $layer \
        --layer_folder CIFAR_top5
done






    