arch="wrn28-2"
dataset="CIFAR10"



for clip in 10 3; do
    python -m src.loss_traces.run_attack_pipeline \
        --exp_id wrn28-2_CIFAR10_c_${clip} \
        --target target \
        --arch $arch \
        --dataset $dataset \
        --target-only \
        --layer 0 \
        --clip_norm $clip 

    python -m src.loss_traces.run_attack_pipeline \
        --exp_id wrn28-2_CIFAR10_c_${clip} \
        --target target \
        --arch $arch \
        --dataset $dataset \
        --shadows-only \
        --layer 0 \
        --clip_norm $clip     
        
    python -m src.loss_traces.run_attack_pipeline \
        --exp_id wrn28-2_CIFAR10_c_${clip} \
        --target target \
        --arch $arch \
        --dataset $dataset \
        --lira-only \
        --layer 0 \
        --clip_norm $clip 
done




    