#!/usr/bin/env bash
# Sequential training template. Add explicit run_script calls at the bottom.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${DEEPKINEMATICS_LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

run_script() {
    local experiment_id=$1
    local model_type=$2
    shift 2
    local additional_args=("$@")
    local timestamp
    timestamp=$(date +%s)
    local log_file="${LOG_DIR}/output_Training_${experiment_id}_${model_type}_${timestamp}.txt"

    "${PYTHON_BIN}" "${PROJECT_ROOT}/DK03_Training/DK03_trainFK.py" \
        --experimentid "${experiment_id}" \
        --m_type "${model_type}" \
        "${additional_args[@]}" > "${log_file}" 2>&1
    echo "Completed ${experiment_id}; log: ${log_file}"
}

# Example (intentionally disabled):
# run_script "FK_example" "rnn" --VERSION FK --use_acc_gyro --n_epochs 100

echo "No sequential jobs are configured. Add run_script calls to this template."
