# Efficiency Lane: Task-Specific Adapters for RoBERTa on AdapterHub

## Overview
Efficiency Lane is a repository that houses task-specific adapters for RoBERTa, structured to facilitate easy data management, model training, and demonstrations of capabilities. Below is a detailed overview of the project's directory structure and functionalities.

## Project Structure

### Code and Processing
** Note, you may want to add `${workspaceFolder}/cs_7643_efficiencylane` to PYTHONPATH, e.g.

`export PYTHONPATH=${PYTHONPATH}:/home/hice1/avizcaino3/scratch/repos/CS-7643-EfficiencyLane/cs_7643_efficiencylane`

- **CS 7643 Efficiency Lane (`./cs_7643_efficiencylane/`)**:
  - **demo/**: Showcases the project capabilities through various demonstrations.
    - Refer to [Link to CS 7643 EfficiencyLane Demo](cs_7643_efficiencylane/demo/README.md) for more detailed instructions on how to train our models and adapters.
  - **utils/**: Contains utilities for mlops to handle our [Optuna](https://github.com/optuna/optuna) trials, data handling and other support functions.
  - **data_loaders/**: Responsible for loading and preprocessing data.
  - **training_configs/**: Holds [Hydra](https://github.com/facebookresearch/hydra) configuration files for training setups.
  - **visualizations**: Custom utility functions made on-the-go for project demands for our analytics layer on model outputs (both for transformers library's mlflow logs, and optuna studies). This is used to produce analysis and plots for our final report.

### Inputs
- **Data (`./data/`)**:
  - Serves as the central repository for datasets. Populated by running `download_datasets.py`, which downloads datasets such as `rct-20k`, `ag`, `chemprot`, `hyperpartisan_news`, `sciie`, `amazon`, `citation_intent`, `imdb`, and `rct-sample`.

#### Downloading Datasets
Execute the following command to download all datasets:
```bash
python cs_7643_efficiencylane/utils/download_datasets.py
```
- **Note**: This process may take several minutes. Modify `datasets.py` to customize which datasets are downloaded. The Amazon reviews dataset, in particular, requires more time due to its size.

### Outputs
- **Adapters (`./adapters/`)**:
  - Output directory for trained adapters, storing model configurations and weights.
  
- **Pretrained Models (`./pretrained_models/`)**:
  - Stores models that have undergone continued pretraining for further use or adaptation.

- **Training Output (`./training_output/`)**:
  - Contains logs and outputs from Optuna training sessions.

- **Analytics Output (`./resources/`)**:
  - Contains plots and tables from our analytics layer.
---

# Demo Descriptions

## 1. Continued Pretraining Replication of Results

These are stored in the `cs_7643_efficiencylane/demo/finetuning` folder. This demo illustrates how to replicate the results of continued pretraining using RoBERTa models. It leverages various utilities and data loaders developed as part of the project.

### Components
- **Data Loader**: Loads the `citation_intent` dataset using the `TaskDataLoader`.
- **Configuration**: Utilizes a `RobertaConfig` with specified dropout probabilities and label count based on the dataset.
- **Model Training**: Executes training using different pretrained model variants such as `allenai/dsp_roberta_base_tapt_citation_intent_1688` to compare performance impacts.
- **Evaluation**: Computes the Macro-F1 score as the primary evaluation metric, highlighting the model's classification performance.

### Execution
To run this demo, set up the environment with necessary libraries, execute the training, and evaluate the model. Configuration details are fetched from YAML files, ensuring that the setup can be easily replicated or adjusted.

You can run commands such as the below to fine-tune pre-trained models (e.g., roberta-base):

  ```bash
  python cs_7643_efficiencylane/demo/finetuning/finetuning.py roberta-base --dataset_name citation_intent --config_name config_finetuning --parallelism 1 --overwrite 1 --job_sequence 1
  ```

## 2. Training New Adapters

These are stored in the `cs_7643_efficiencylane/demo/adapters` folder. We train new adapters showcasing the flexibility and power of the adapter-based approach in NLP.

### Components
- **Adapter Training Setup**: Involves initializing a RoBERTa model with an added adapter and a matching classification head for the specific dataset.
- **Optuna Integration**: Utilizes Optuna for hyperparameter tuning to optimize the training process, focusing on learning rates and batch sizes.
- **Adapter Fusion**: A POC is build that supports the fuse of two distinct adapters, showcasing the multi-task and transfer learning capabilities of adapters.

### Execution
To execute this demo, ensure the model, adapter, and training configurations are correctly set up. The training process is managed by the `AdapterTrainer`, which focuses solely on adapting the newly added components. After training, the adapter can be saved locally or pushed to the [Adapter-Hub](https://adapterhub.ml/) for broader accessibility.

Similar to finetuning, you can run commands such as the below to train adapters.

    ```bash
    python cs_7643_efficiencylane/demo/adapters/training_adapters.py roberta-base --dataset_name citation_intent --adapter_config_name seq_bn --study_suffix adapter_fusion_test --config_path ../../training_configs --config_name roberta-base --parallelism 0 --overwrite 1 --job_sequence 1
    ```


## 3. Logs and Visualisation
Web version of optuna dashboard is more complete than VSCode version:
optuna-dashboard sqlite:///db.sqlite3

We publish our code for analytics layer, but this is for reference only; do not expect it to be maintained or work for your use-case.
