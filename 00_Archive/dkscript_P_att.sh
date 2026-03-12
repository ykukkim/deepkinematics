#!/bin/bash

pids=""
index=0

# Function to run a python script and log the job ID

run_script() {
    local script_name=$1
    local experiment_id=$2
    local m_type=$3
    # Shift the first three arguments to correctly capture additional arguments
    shift 3
    local additional_args=$@

    # Get the current timestamp
    local timestamp=$(date +%s)

    # Incremental index for the log files
    index=$((index + 1))

    # Define log file names using the index
    local log_file="$HOME/output_Training_${experiment_id}_${m_type}_${timestamp}_${index}.txt"

    # Start the python script in the background, redirecting both stdout and stderr to the same log file
    python "$HOME/dl_humanmotion/$script_name" --experimentid "${experiment_id}"  --m_type "${m_type}" ${additional_args} > "${log_file}" 2>&1 &

    local pid=$!

    echo "Script with experiment ID ${experiment_id} started with PID ${pid}"
    echo "${experiment_id}: ${pid}" >> job_ids.log

    # Store the PID in the string
    pids="${pids} ${pid}"
}

# Running scripts simultaneously with different parameters
run_script "DK03_trainFK_Euler.py" "FK_full_att_NS3" "att" --VERSION 'FK' --bs_train 64 --n_epochs 200 --eval_every 10 --optimizer 'adam' --scheduler 'reduce' --lr 5e-3 --weight_decay 5e-3 --m_dropout 0.3 --m_pose_loss 10 --m_joint_rot_loss 20 --m_num_layers_attention 2 --m_num_hidden_units_attention '[[512,512]]' --m_num_heads_attention '[[16,16],[32,32]]' --m_embedding_attention '[[128,128]]' --m_embedding_positional '[[32]]' --m_window_attention 10 --m_skip_connections --use_acc_gyro --use_acc_gyro --use_batch_norm --predict_arms --predict_head --predict_contact --predict_spl
# Wait for all background processes to complete
for pid in ${pids}; do
    wait $pid
done

echo "All Python scripts completed."