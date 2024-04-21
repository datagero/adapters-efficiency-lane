from env.logger_config import get_logger
logger = get_logger()

import argparse
import torch
import hydra
from transformers import RobertaConfig, RobertaForSequenceClassification
from hydra import initialize, compose

# Our built utilities
from data_loaders.citation_intent_data_loader import CSTasksDataLoader
from utils import optuna_objectives


def get_model_and_data(model_variant, dataset_name):
    # ======================================================
    # Set-up and Load Data
    # ======================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = CSTasksDataLoader(model_name="roberta-base",
                                    dataset_name=dataset_name,
                                    path=f"data/{dataset_name}/",
                                    checkpoint_path=f"data/{dataset_name}/processed_dataset.pt")

    dataset = loader.load_dataset(overwrite=False)

    # ======================================================
    # Model Config & Training
    # ======================================================
    # Set up training for the Model and Adapter
    config = RobertaConfig.from_pretrained(
        'roberta-base',
        num_labels=loader.num_labels,
        problem_type="single_label_classification",
        hidden_dropout_prob=0.1,
    )

    model = RobertaForSequenceClassification.from_pretrained(model_variant, config=config)
    model.to(device)

    return model, dataset

if __name__ == "__main__":

    """
    Run different pre-trained models by changing the model_variant variable.
    We will then build a classification head for the model.
    e.g. 
        roberta-base is the base pre-trained model.
        allenai/cs_roberta_base is the base pre-trained model.
        ./mlm_model is our pre-trained with tapt data.
        allenai/dsp_roberta_base_tapt_citation_intent_1688 is the published (2020) pre-trained with tapt data.
        allenai/dsp_roberta_base_dapt_cs_tapt_citation_intent_1688 is the published (2020) pre-trained with dapt and then with tapt data.
    """
    parser = argparse.ArgumentParser(description='Training Classifier Head for pre-trained model.')
    parser.add_argument('model_variant', type=str, default='roberta-base', help='the model variant to use (default: roberta-base)')
    parser.add_argument('--dataset_name', type=str, default='citation_intent', help='the name of the dataset')
    parser.add_argument('--study_suffix', type=str, default='default_test', help='the suffix to add to the study name')
    parser.add_argument('--config_path', type=str, default='../../training_configs', help='the path to training configuration files')
    parser.add_argument('--config_name', type=str, default='classifier_head_test', help='the name of the configuration file')
    parser.add_argument('--job_sequence', type=int, default=1, help='the number of job for parallel runs (default: 1)')
    args = parser.parse_args()

    model_variant = args.model_variant
    dataset_name = args.dataset_name
    config_path = args.config_path
    config_name = args.config_name
    study_suffix = args.study_suffix
    job_sequence = args.job_sequence

    logger.info(f"Inputs: {model_variant}, {dataset_name}, {config_path}, {config_name}, {study_suffix}")

    # print(f"Starting training for Model Variant: {model_variant} with Config: {config_name} loaded from {config_path}")
    logger.info(f"Starting training for Model Variant: {model_variant} with Config: {config_name} loaded from {config_path}")
    
    save_model_name = model_variant.split("/")[-1] + "_" + dataset_name

    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)
        
    model, dataset = get_model_and_data(model_variant, dataset_name)
    study_name = f"{save_model_name}_training_{study_suffix}"

    # Run in a single thread. You could run this file multiple times for the same study, Optuna would manage parallelism:
    optuna_objectives.run_study_experiments(cfg, model=model, dataset=dataset, study_name=study_name, job_sequence=job_sequence)
