#!/bin/bash
# This script provides a robust, one-shot method to launch the entire training process.
# It handles dependencies, configuration, and execution, ensuring no manual setup is needed.

set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================================"
echo "ProbSolSpace 1.0 Transformer Experimental: Launch Sequence Start"
echo "================================================================"

# --- Step 1: Environment Sanity Checks ---
echo "[1/5] Performing environment sanity checks..."
if ! command -v nvidia-smi &> /dev/null
then
    echo "ERROR: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    exit 1
fi
if ! command -v python &> /dev/null
then
    echo "ERROR: Python not found."
    exit 1
fi
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "WARNING: Python version is $PYTHON_VERSION. This script is tested on 3.10. Proceeding with caution."
fi
echo "Environment checks passed."

# --- Step 2: Install Dependencies ---
echo "[2/5] Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed."

# --- Step 3: Configure Hugging Face Accelerate ---
echo "[3/5] Configuring Hugging Face Accelerate for FSDP..."
# This command non-interactively creates the accelerate_config.yaml file.
# The settings are optimized for 4x A100s on a single machine.
# PASTE THIS NEW BLOCK
cat <<EOF > accelerate_config.yaml
# Configuration for Hugging Face Accelerate with FSDP on 4 A100 GPUs
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_use_cluster: false
use_cpu: false
EOF
echo "Accelerate configured. Config file 'accelerate_config.yaml' created/updated."
cat accelerate_config.yaml

# --- Step 4: Create Results Directory ---
echo "[4/5] Creating results directory..."
mkdir -p results
echo "Results directory ready."

# --- Step 5: Launch Distributed Training ---
echo "[5/5] Launching the main training script with Accelerate..."
echo "All systems go. Initiating training run. This may take several hours."
echo "Follow the logs for real-time progress..."
echo "================================================================"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch -m training.trainer

echo "================================================================"
echo "ProbSolSpace 1.0 Training Run COMPLETE."
echo "Final model saved to: ./results/final_model"
echo "Training report saved to: ./results/training_report.png"
echo "================================================================"
