arch="wrn28-2"
dataset="CIFAR10"


for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
    echo "Running layer $layer"
    python -m src.loss_traces.run_attack_pipeline \
        --exp_id CIFAR_top2_l${layer} \
        --target target \
        --arch $arch \
        --dataset $dataset \
        --target-only \
        --layer $layer \
        --layer_folder CIFAR_top2
        
    python src/privacy_onion/layer_utils.py \
        --exp_id CIFAR_top2_l${layer} \
        --layer $layer \
        --exp_path CIFAR_top2 \
        --top_k 0.05

    echo "Training shadows for layer $layer"
    python -m src.loss_traces.run_attack_pipeline \
        --exp_id CIFAR_top2_l${layer} \
        --target target \
        --arch $arch \
        --dataset $dataset \
        --shadows-only \
        --layer $layer \
        --layer_folder CIFAR_top2
    
    python -m src.loss_traces.run_attack_pipeline \
        --exp_id CIFAR_top2_l${layer} \
        --target target \
        --arch $arch \
        --dataset $dataset \
        --lira-only \
        --layer $layer \
        --layer_folder CIFAR_top2
done






    