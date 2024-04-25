#!/bin/bash

# Example run:
# bash cs_7643_efficiencylane/utils/run_parallel_adapter.sh roberta-base citation_intent pfeiffer base_v01

# Example kill
# ps aux --sort=-%cpu | head -n 6
# or
# ps aux | grep "cs_7643_efficiencylane/utils/run_parallel.sh"
# pkill -P <parent_id>
# kill <process_id>

# Number of times to run the script in parallel 
# (i.e., enables Optuna trials in parallel)
NUM_RUNS=1

# Define the model_variant argument
MODEL_VARIANT=$1
DATASET_NAME=$2
ADAPTER_CONFIG_NAME=$3
CONFIG_NAME=$4
STUDY_SUFFIX=$5
PARALLELISM=${6:0}
OVERWRITE=${7:0}

for i in $(seq 1 $NUM_RUNS)
do
    # Pass the model_variant argument to the Python script
    python cs_7643_efficiencylane/demo/adapters/new_adapter.py \
        "$MODEL_VARIANT" \
        --dataset_name "$DATASET_NAME" \
        --adapter_config_name "$ADAPTER_CONFIG_NAME" \
        --config_name "$CONFIG_NAME" \
        --study_suffix "$STUDY_SUFFIX" \
        --parallelism "$PARALLELISM" \
        --overwrite "$OVERWRITE" \
        --job_sequence $i &
    sleep 2 # Sleep for 2 seconds before starting the next job
done

wait # Wait for all background jobs to finish
echo "All parallel scripts have finished executing."
