source /scratch/euodia/myenv/bin/activate
log_path=/scratch/euodia/logs
norm_tracker_path=/home/euodia/loss_traces

exp_id="test"
cpus="19-20"
gpu=":0"

dataset="CIFAR10"
arch="simple_convnet"
epochs=20
lr=0.05
bs=64

shadow_count=256
dual_count=1


echo "Took $elapsed_time seconds"
for ((i=0; i<$dual_count; i++))
do
    echo "Starting attack-r dual ${i}"
    taskset -c $cpus python $norm_tracker_path/attacks/attack_r.py --gpu $gpu --exp_id $exp_id --target_id dual_track_both_$i --batchsize $bs
done