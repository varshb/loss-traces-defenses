arch="wrn28-2"
dataset="CIFAR10"



for clip in 3.0 10.0; do
    for layer in 5 10 20 30 40; do
        python src/privacy_onion/layer_utils.py \
            --exp_id CIFAR_l0 \
            --layer 0 \
            --exp_path selective_clipping_${layer} \
            --method top_k \
            --top_k echo "scale=2; $layer/100" | bc
        
        python -m src.loss_traces.run_attack_pipeline  \
            --exp_id CIFAR_selective_l${layer}_${clip}   \
            --arch $arch   \
            --dataset $dataset  \
            --layer 1  \
            --target-only  \
            --clip_norm ${clip}  \
            --selective_clip   \
            --layer_folder selective_clipping_${layer}  

        python -m src.loss_traces.run_attack_pipeline  \
            --exp_id CIFAR_selective_l${layer}_${clip}   \
            --arch $arch   \
            --dataset $dataset  \
            --layer 1  \
            --shadows-only  \
            --clip_norm ${clip}  \
            --selective_clip   \
            --layer_folder selective_clipping_${layer}  

        python -m src.loss_traces.run_attack_pipeline  \
            --exp_id CIFAR_selective_l${layer}_${clip}   \
            --arch $arch   \
            --dataset $dataset  \
            --layer 0  \
            --lira-only  
    done
done
