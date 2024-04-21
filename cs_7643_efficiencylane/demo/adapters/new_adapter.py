"""
This was initially inspired from https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb#scrollTo=huLjPAKHLA1g
"""
from env.logger_config import get_logger
logger = get_logger()

import argparse
import os
import torch
import hydra
from omegaconf import DictConfig 
from adapters import RobertaAdapterModel, AdapterConfig
from transformers import RobertaConfig
from transformers import TextClassificationPipeline
from hydra import initialize, compose

# Our built utilities
from data_loaders.citation_intent_data_loader import CSTasksDataLoader
from utils import compute_metrics, optuna_objectives

def get_model_and_data(model_variant, dataset_name, adapter_config_name):
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
        "roberta-base",
        num_labels=loader.num_labels,
    )

    model = RobertaAdapterModel.from_pretrained(model_variant, config=config)
    model.to(device)
    
    # Add a new adapter and a matching classification head
    adapter_name = model_variant.split("/")[-1]+"_"+dataset_name+"_"+adapter_config_name
    print(f"Adapter Name: {adapter_name}")
    
    model.add_adapter(adapter_name, config=adapter_config_name)
    model.add_classification_head(
        adapter_name,
        num_labels=loader.num_labels
    )

    #  The train_adapter() method does two things:
    #     It freezes all weights of the pre-trained model, so only the adapter weights are updated during training.
    #     It activates the adapter and the prediction head such that both are used in every forward pass.
    # Activate the adapter
    model.train_adapter(adapter_name)

    return model, dataset, adapter_name


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
    parser.add_argument('model_variant', type=str, default='allenai/cs_roberta_base', help='the model variant to use (default: roberta-base)')
    parser.add_argument('--dataset_name', type=str, default='citation_intent', help='the name of the dataset')
    parser.add_argument('--adapter_config_name', type=str, default='seq_bn', help='the name of the adapter configuration file')
    parser.add_argument('--study_suffix', type=str, default='default_test', help='the suffix to add to the study name')
    parser.add_argument('--config_path', type=str, default='../../training_configs', help='the path to training configuration files')
    parser.add_argument('--config_name', type=str, default='adapter_citation_intent_test', help='the name of the configuration file')
    parser.add_argument('--job_sequence', type=int, default=1, help='the number of job for parallel runs (default: 1)')
    # To add support for different databases
    args = parser.parse_args()

    model_variant = args.model_variant
    dataset_name = args.dataset_name
    adapter_config_name = args.adapter_config_name
    config_path = args.config_path
    config_name = args.config_name
    study_suffix = args.study_suffix
    job_sequence = args.job_sequence

    logger.info(f"Inputs: {model_variant}, {dataset_name}, {adapter_config_name}, {config_path}, {config_name}, {study_suffix}")

    # print(f"Starting training for Model Variant: {model_variant} with Config: {config_name} loaded from {config_path}")
    logger.info(f"Starting training for Model Variant: {model_variant} with Config: {config_name} loaded from {config_path}")
    
    save_model_name = model_variant.split("/")[-1] + '_' + dataset_name + '_' + adapter_config_name

    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    model, dataset, adapter_name = get_model_and_data(model_variant, dataset_name, adapter_config_name)
    study_name = f"{save_model_name}_training_{study_suffix}"

    # Run in a single thread. You could run this file multiple times for the same study, Optuna would manage parallelism:
    optuna_objectives.run_study_experiments(cfg, model=model, dataset=dataset, study_name=study_name, trainer_type='adapter', job_sequence=job_sequence)

    # # Test the model
    # classifier = TextClassificationPipeline(model=model, tokenizer=loader.tokenizer, device=0)
    # test_result = classifier("We use the same set of binary features as in previous work on this dataset ( Pang et al. , 2002 ; Pang and Lee , 2004 ; Zaidan et al. , 2007 ) .")
    # print(test_result)

    out_fldr_base = "./adapters"
    out_fldr = f"{out_fldr_base}/{model_variant}/{dataset_name}/{adapter_config_name}"
    if not os.path.exists(out_fldr):
        os.makedirs(out_fldr)

    model.save_adapter(out_fldr, adapter_name)

