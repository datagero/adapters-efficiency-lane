#!/bin/bash

#SBATCH -AdapterTraining                    # Job name
#SBATCH -N1 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=52G                        # Memory per core
#SBATCH --time=16:00:00                         # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=avizcaino3@gatech.edu        # E-mail address for notifications

cd /home/hice1/avizcaino3/scratch/repos/CS-7643-EfficiencyLane

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
    srun python cs_7643_efficiencylane/demo/adapters/new_adapter.py \
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
