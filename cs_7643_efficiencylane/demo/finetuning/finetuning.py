from env.logger_config import get_logger
logger = get_logger()

import os
import argparse
from hydra import initialize, compose

# Our built utilities
from utils import optuna_objectives

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
    parser.add_argument('--config_name', type=str, default='finetuning_test', help='the name of the configuration file')
    parser.add_argument('--parallelism', type=bool, default=True, help='the number of job for parallel runs (default: 1)')
    parser.add_argument('--job_sequence', type=int, default=1, help='the number of job for parallel runs (default: 1)')
    args = parser.parse_args()

    model_variant = args.model_variant
    dataset_name = args.dataset_name
    config_path = args.config_path
    config_name = args.config_name
    study_suffix = args.study_suffix
    paralellism = args.parallelism
    job_sequence = args.job_sequence

    study_config_path = os.path.join(config_path, study_suffix)

    logger.info(f"Inputs: {model_variant}, {dataset_name}, {config_path}, {config_name}, {study_suffix}")

    # print(f"Starting training for Model Variant: {model_variant} with Config: {config_name} loaded from {config_path}")
    logger.info(f"Starting training for Model Variant: {model_variant} with Config: {config_name} loaded from {study_config_path}")

    with initialize(config_path=study_config_path):
        cfg = compose(config_name=config_name)
        
    save_model_name = model_variant.split("/")[-1] + "_" + dataset_name
    study_name = f"{save_model_name}_training_{study_suffix}"

    run_study_args = {
        'model_variant': model_variant,
        'trainer_type': 'model',
        'dataset_name': dataset_name,
        'study_name': study_name,
        'job_sequence': job_sequence,
        'parallelism': paralellism
    }

    # Run in a single thread. You could run this file multiple times for the same study, Optuna would manage parallelism:
    optuna_objectives.run_study_experiments(cfg, **run_study_args)
