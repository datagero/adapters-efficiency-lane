# DEMO - Transformer's Finetuning and Adapter Training

This module provides tools for training deep learning models, focusing on both using adapters with pre-trained models and fine-tuning models directly. It leverages Hydra configuration management library and Optuna for hyperparameter optimization. Below are detailed instructions on how to set up and use this module for both training scenarios.

## Requirements

As described on poetry's pyproject.tom

Note, you may want to add `${workspaceFolder}/cs_7643_efficiencylane` to PYTHONPATH
e.g.,

`export PYTHONPATH=${PYTHONPATH}:/home/hice1/avizcaino3/scratch/repos/CS-7643-EfficiencyLane/cs_7643_efficiencylane`

```bash
pip install hydra-core optuna torch transformers
```

## Configuration

Before running the training scripts, ensure you configure the environment and dependencies correctly:

1. **Logger Configuration**: Ensure that the `env/logger_config.py` script is set up to output logs according to your requirements.
2. **Data Downloads**: Make sure your datasets are downloaded by running python `cs_7643_efficiencylane/utils/download_datasets.py`


## Scripts Description

- **Training with Adapters**: `adapters/training_adapters.py`
- **Fine-Tuning**: `finetuning/finetuning.py`

### Training with Adapters

This script allows the training of adapter modules on top of pre-trained models. It is ideal for experiments where you want to retain the original pre-trained model weights untouched while adapting to new tasks.

#### Usage

To run the adapter training script, you will need to specify various parameters such as the model variant and the dataset name. Below is an example command:

```bash
python cs_7643_efficiencylane/demo/adapters/training_adapters.py roberta-base --dataset_name citation_intent --adapter_config_name seq_bn --study_suffix adapter_fusion_test --config_path ../../training_configs --config_name roberta-base --parallelism 0 --overwrite 1 --job_sequence 1
```

### Fine-Tuning

This script is used for direct fine-tuning of pre-trained models on specific tasks or datasets, allowing modifications to the original model weights.

#### Usage

Similar to the adapter training script, you will specify the model and dataset among other configurations. An example command for running the fine-tuning script is shown below:

```bash
python cs_7643_efficiencylane/demo/finetuning/finetuning.py roberta-base --dataset_name citation_intent --config_name config_finetuning --parallelism 1 --overwrite 1 --job_sequence 1
```

## Parallel Execution

Both scripts support parallel execution to expedite the training process, in the case where multiple seeds are required per experiment. This is controlled via the `--parallelism` flag, which, when set to `1`, allows the scripts to manage multiple jobs in parallel, optimizing hyperparameters across different settings concurrently. Note, that if there is insufficient memory, then the process will error.

## Additional Notes

- **Hydra Configurations**: Configuration files specified in `--config_path` should be set up according to the Hydra documentation.
- **Optuna Studies**: Hyperparameter studies are managed via Optuna, with results stored in a SQLite database. Ensure the database path is accessible.

---