# Efficiency Lane: Task-Specific Adapters for RoBERTa on AdapterHub

** Note, you may want to add `${workspaceFolder}/cs_7643_efficiencylane` to PYTHONPATH
e.g.,

`export PYTHONPATH=${PYTHONPATH}:/home/hice1/avizcaino3/scratch/repos/CS-7643-EfficiencyLane/cs_7643_efficiencylane`

## Overview
Efficiency Lane is a repository that houses task-specific adapters for RoBERTa, structured to facilitate easy data management, model training, and demonstrations of capabilities. Below is a detailed overview of the project's directory structure and functionalities.

## Project Structure

### Code and Processing
- **CS 7643 Efficiency Lane (`./cs_7643_efficiencylane/`)**:
  - **demo/**: Showcases the project capabilities through various demonstrations.
  - **utils/**: Contains utilities for data handling and other support functions.
  - **data_loaders/**: Responsible for loading and preprocessing data.
  - **training_configs/**: Holds configuration files for training setups.

### Inputs
- **Data (`./data/`)**:
  - Serves as the central repository for datasets. Populated by running `download_datasets.py`, which downloads datasets such as `rct-20k`, `ag`, `chemprot`, `hyperpartisan_news`, `sciie`, `amazon`, `citation_intent`, `imdb`, and `rct-sample`.

#### Downloading Datasets
Execute the following command to download all datasets:
```bash
python download_datasets.py
```
- **Note**: This process may take several minutes. Modify `datasets.py` to customize which datasets are downloaded. The Amazon reviews dataset, in particular, requires more time due to its size.

### Outputs
- **Adapters (`./adapters/`)**:
  - Output directory for trained adapters, storing model configurations and weights.
  
- **Pretrained Models (`./pretrained_models/`)**:
  - Stores models that have undergone continued pretraining for further use or adaptation.

- **Training Output (`./training_output/`)**:
  - Contains logs and outputs from Optuna training sessions.



---

# Demo Descriptions

## 1. Continued Pretraining Replication of Results

These are stored in the `cs_7643_efficiencylane/demo/finetuning` folder. This demo illustrates how to replicate the results of continued pretraining using RoBERTa models. It leverages various utilities and data loaders developed as part of the project.

### Components
- **Data Loader**: Loads the `citation_intent` dataset using the `CitationIntentDataLoader`.
- **Configuration**: Utilizes a `RobertaConfig` with specified dropout probabilities and label count based on the dataset.
- **Model Training**: Executes training using different pretrained model variants such as `allenai/dsp_roberta_base_tapt_citation_intent_1688` to compare performance impacts.
- **Evaluation**: Computes the Macro-F1 score as the primary evaluation metric, highlighting the model's classification performance.

### Execution
To run this demo, set up the environment with necessary libraries, execute the training, and evaluate the model. Configuration details are fetched from YAML files, ensuring that the setup can be easily replicated or adjusted.

## 2. Training New Adapters

These are stored in the `cs_7643_efficiencylane/demo/adapters` folder. We train new adapters showcasing the flexibility and power of the adapter-based approach in NLP.

### Components
- **Adapter Training Setup**: Involves initializing a RoBERTa model with an added adapter and a matching classification head for the specific dataset.
- **Optuna Integration**: Utilizes Optuna for hyperparameter tuning to optimize the training process, focusing on learning rates and batch sizes.
- **Pipeline Testing**: A `TextClassificationPipeline` is used to test the adapter's performance on sample text, ensuring that the model functions as expected after training.

### Execution
To execute this demo, ensure the model, adapter, and training configurations are correctly set up. The training process is managed by the `AdapterTrainer`, which focuses solely on adapting the newly added components. After training, the adapter can be saved locally or pushed to the Adapter-Hub for broader accessibility.


## 3. Logs and Visualisation
Web version of optuna dashboard is more complete than VSCode version:
optuna-dashboard sqlite:///db.sqlite3

To delete experiments from optuna:
sqlite3 db.sqlite3
sqlite> DELETE FROM studies WHERE study_name LIKE 'cs_roberta_base_training_default%';
