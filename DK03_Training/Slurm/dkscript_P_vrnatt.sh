#!/usr/bin/env bash
# Parallel velocity-plus-attention training example.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${DEEPKINEMATICS_LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

pids=""
index=0

# Function to run a python script and log the job ID

run_script() {
    local experiment_id=$1
    local m_type=$2
    shift 2
    local additional_args=("$@")

    # Get the current timestamp
    local timestamp=$(date +%s)

    # Incremental index for the log files
    index=$((index + 1))

    # Define log file names using the index
    local log_file="${LOG_DIR}/output_Training_${experiment_id}_${m_type}_${timestamp}_${index}.txt"

    # Start the python script in the background, redirecting both stdout and stderr to the same log file
    "${PYTHON_BIN}" "${PROJECT_ROOT}/DK03_Training/DK03_trainFK.py" \
        --experimentid "${experiment_id}" --m_type "${m_type}" \
        "${additional_args[@]}" > "${log_file}" 2>&1 &

    local pid=$!

    echo "Script with experiment ID ${experiment_id} started with PID ${pid}"
    echo "${experiment_id}: ${pid}" >> job_ids.log

    # Store the PID in the string
    pids="${pids} ${pid}"
}

# Running scripts simultaneously with different parameters
run_script "FK_search_vrnatt" "vrnatt" --VERSION 'FK' --bs_train 64 --n_epochs 200 --eval_every 10 --optimizer 'adam' --scheduler 'reduce' --lr 1e-3 --weight_decay 1e-5 --m_dropout 0.3 --m_pose_loss 10 --m_joint_rot_loss 100 --m_num_layers_VRN 1 --m_hidden_units_VRN '[[256],[1024]]' --m_embedding_VRNMLP '[[32],[512]]' --m_bidirectional --m_learn_init_state --m_positional_encoding_type sinusoidal --m_window_positional 100 500  --m_embedding_ATTMLP '[[16],[16,32]]' --m_num_layers_attention 2 --m_num_hidden_units_attention '[[128,128],[1024,1024]]' --m_num_heads_attention '[[8,8],[32,32]]' --m_embedding_attention '[[32,32],[256,256]]' --m_window_attention 100 500 --m_skip_connections --use_acc_gyro --predict_arms --predict_head --predict_contact --predict_spl

# Wait for all background processes to complete
for pid in ${pids}; do
    wait $pid
done

echo "All Python scripts completed."
