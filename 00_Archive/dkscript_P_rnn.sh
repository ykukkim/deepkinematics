#!/bin/bash

pids=""
index=0

# Function to run a python script and log the job ID

run_script() {
    local script_name=$1
    local experiment_id=$2
    local m_type=$3
    shift 3    # Shift the first three arguments to correctly capture additional arguments

    # Capture all additional arguments as an array to preserve quoting
    local additional_args=("$@")
    local timestamp=$(date +%s)
    index=$((index + 1))
    local log_file="$HOME/output_Training_${experiment_id}_${m_type}_${timestamp}_${index}.txt"

    # Start the python script in the background, redirecting both stdout and stderr to the same log file
    python "$HOME/dl_humanmotion/$script_name" \
        --experimentid "${experiment_id}" \
        --m_type "${m_type}" \
        "${additional_args[@]}" > "${log_file}" 2>&1 &

    local pid=$!
    echo "Script with experiment ID ${experiment_id} started with PID ${pid}"
    echo "${experiment_id}: ${pid}" >> "job_ids_${experiment_id}_${timestamp}".log
    pids="${pids} ${pid}"
}

# Running scripts simultaneously with different parameters
run_script "DK03_trainFK_Euler.py" "FK_Full_Adam_reduce" 'rnn' --VERSION 'FK' --bs_train 64 --n_epochs 200 --eval_every 10 --optimizer 'adam' --scheduler 'reduce' --lr 1e-3 --weight_decay 1e-6 --m_dropout 0.3 --m_pose_loss 10 --m_joint_rot_loss 20 --m_num_layers_RNN 2  --m_hidden_units_RNN '[[1024]]' --m_bidirectional --m_learn_init_state --use_acc_gyro --use_batch_norm --predict_arms --predict_head --predict_contact --predict_spl

for pid in ${pids}; do
    wait $pid
done

echo "All Python scripts completed."