#!/bin/bash

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
    python "$HOME/dl_humanmotion/$script_name" --experimentid "${experiment_id}"  --m_type "${m_type}" ${additional_args} > "${log_file}" 2>&1

    local pid=$!

    echo "Script with experiment ID ${experiment_id} started with PID ${pid}"
    echo "${experiment_id}: ${pid}" >> job_ids.log

    # Store the PID in the string
    pids="${pids} ${pid}"
}


# Running scripts sequentially with different parameters
#run_script "DK03_trainFK_Att.py" "FK_test_att" "vrnatt" --VERSION 'FK' --bs_train 128 --n_epochs 100 --eval_every 5 --optimizer 'adam' --scheduler 'reduce' --lr 5e-2 --weight_decay 1e-3 --m_dropout 0.3 --m_pose_loss 40 --m_joint_rot_loss 100 --m_contact_loss 0.1 --m_orientation_loss 1 --m_num_layers 1 --m_hidden_units '[[512]]' --m_embedding '[[8]]' --m_num_layers_attention 1 --m_num_hidden_units_attention '[[512]]' --m_num_heads_attention '[[2],[8]]' --m_embedding_attention '[[8]]' --m_embedding_positional '[[8]]' --m_window_attention 10 --m_skip_connections --use_batch_norm --use_acc_gyro
#run_script "DK03_trainFK_Att.py" "FK_test_att" "vrnatt" --VERSION 'FK' --bs_train 128 --n_epochs 100 --eval_every 5 --optimizer 'adam' --scheduler 'reduce' --lr 5e-2 --weight_decay 1e-3 --m_dropout 0.3 --m_pose_loss 40 --m_joint_rot_loss 100 --m_contact_loss 0.1 --m_orientation_loss 1 --m_num_layers 1 --m_hidden_units '[[32]]' --m_embedding '[[8]]' --m_num_layers_attention 1 --m_num_hidden_units_attention '[[512]]' --m_num_heads_attention '[[2],[8]]' --m_embedding_attention '[[8]]' --m_embedding_positional '[[8]]' --m_window_attention 10 --m_skip_connections --use_batch_norm --use_acc_gyro
#run_script "DK03_trainFK_Att.py" "FK_test_att" "vrnatt" --VERSION 'FK' --bs_train 128 --n_epochs 100 --eval_every 5 --optimizer 'adam' --scheduler 'reduce' --lr 5e-2 --weight_decay 1e-3 --m_dropout 0.3 --m_pose_loss 40 --m_joint_rot_loss 100 --m_contact_loss 0.1 --m_orientation_loss 1 --m_num_layers 1 --m_hidden_units '[[32]]' --m_embedding '[[8]]' --m_num_layers_attention 1 --m_num_hidden_units_attention '[[512]]' --m_num_heads_attention '[[2],[8]]' --m_embedding_attention '[[8]]' --m_embedding_positional '[[8]]' --m_window_attention 10 --m_skip_connections --use_batch_norm --use_acc_gyro
#run_script "DK03_trainFK_Att.py" "FK_test_att" "vrnatt" --VERSION 'FK' --bs_train 128 --n_epochs 100 --eval_every 5 --optimizer 'adam' --scheduler 'reduce' --lr 5e-2 --weight_decay 1e-3 --m_dropout 0.3 --m_pose_loss 40 --m_joint_rot_loss 100 --m_contact_loss 0.1 --m_orientation_loss 1 --m_num_layers 1 --m_hidden_units '[[32]]' --m_embedding '[[8]]' --m_num_layers_attention 1 --m_num_hidden_units_attention '[[32]]' --m_num_heads_attention '[[2],[8]]' --m_embedding_attention '[[8]]' --m_embedding_positional '[[8]]' --m_window_attention 10 --m_skip_connections --use_batch_norm --use_acc_gyro
#run_script "DK03_trainFK_Att.py" "FK_test_att" "vrnatt" --VERSION 'FK' --bs_train 128 --n_epochs 100 --eval_every 5 --optimizer 'adam' --scheduler 'reduce' --lr 5e-2 --weight_decay 1e-3 --m_dropout 0.3 --m_pose_loss 40 --m_joint_rot_loss 100 --m_contact_loss 0.1 --m_orientation_loss 1 --m_num_layers 1 --m_hidden_units '[[2048]]' --m_embedding '[[1024]]' --m_num_layers_attention 1 --m_num_hidden_units_attention '[[512]]' --m_num_heads_attention '[[2],[8]]' --m_embedding_attention '[[1024]]' --m_embedding_positional '[[1024]]' --m_window_attention 10 --m_skip_connections --use_batch_norm --use_acc_gyro
#run_script "DK03_trainFK_Att.py" "FK_test_att" "vrnatt" --VERSION 'FK' --bs_train 128 --n_epochs 100 --eval_every 5 --optimizer 'adam' --scheduler 'reduce' --lr 5e-2 --weight_decay 1e-3 --m_dropout 0.3 --m_pose_loss 40 --m_joint_rot_loss 100 --m_contact_loss 0.1 --m_orientation_loss 1 --m_num_layers 1 --m_hidden_units '[[32]]' --m_embedding '[[8]]' --m_num_layers_attention 1 --m_num_hidden_units_attention '[[512]]' --m_num_heads_attention '[[2],[8]]' --m_embedding_attention '[[8]]' --m_embedding_positional '[[1024]]' --m_window_attention 10 --m_skip_connections --use_batch_norm --use_acc_gyro

# Wait for all background processes to complete
for pid in ${pids}; do
    wait $pid
done

echo "All Python scripts completed."


