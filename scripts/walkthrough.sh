exp_id="test"
cpus="19-20"
gpu=":0"

dataset="CIFAR10"
arch="wrn28-2"
epochs=1
lr=0.05
bs=64

seed=1234

shadow_count=256
dual_count=10


# If augmenting we use --track_computed_loss, otherwise --track_free_loss

# Train 1st model 
echo "Training target model"
start_time=$(date +%s)
taskset -c $cpus python main.py --track_computed_loss --augment --gpu $gpu --dataset $dataset --seed $seed --arch $arch --batchsize $bs --lr $lr --epochs $epochs --exp_id $exp_id
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Took $elapsed_time seconds"

## Train 'dual' models on same training set, varying randomness
for ((i=0; i<$dual_count; i++))
do
    echo "Training dual model $i"
    start_time=$(date +%s) #
    taskset -c $cpus python main.py --track_computed_loss --augment --gpu $gpu --dataset $dataset --seed ${seed}${i} --dual track_both_$i --arch $arch --batchsize $bs --lr $lr --epochs $epochs --exp_id $exp_id
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Took $elapsed_time seconds"
done

# Train shadow models
echo "Training shadow models"
start_time=$(date +%s)
taskset -c $cpus python main.py --augment --model_start 0 --model_stop $shadow_count --gpu $gpu --dataset $dataset --arch $arch --batchsize $bs --lr $lr --epochs $epochs --shadow_count $shadow_count --exp_id $exp_id
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Took $elapsed_time seconds"

## Then run attacks