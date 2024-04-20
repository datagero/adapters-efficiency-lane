#!/bin/bash

# Example run:
# bash cs_7643_efficiencylane/utils/run_parallel.sh allenai/cs_roberta_base base_v01

# Example kill
# ps aux --sort=-%cpu | head -n 6
# or
# ps aux | grep "cs_7643_efficiencylane/utils/run_parallel.sh"
# pkill -P <parent_id>
# kill <process_id>

# Number of times to run the script in parallel
NUM_RUNS=3

# Define the model_variant argument
MODEL_VARIANT=$1
STUDY_SUFFIX=$2

for i in $(seq 1 $NUM_RUNS)
do
    # Pass the model_variant argument to the Python script
    python cs_7643_efficiencylane/demo/finetuning/classifier_head.py \
        "$MODEL_VARIANT" \
        --study_suffix "$STUDY_SUFFIX" &
    sleep 2 # Sleep for 2 seconds before starting the next job
done

wait # Wait for all background jobs to finish
echo "All parallel scripts have finished executing."
