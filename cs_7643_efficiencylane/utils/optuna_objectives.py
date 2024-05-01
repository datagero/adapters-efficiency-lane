"""
Here we define the Optuna objective and trigger our training processes.
The results from the seeds get averaged and the average is used as the final result for the trial.
Logs are written in consistent form to feed other project modules and further analysis/visualization.
Optuna also maintains its own dashboard the we can access through a GUI.
Experimental - support parallelism for different random seeds.
"""
from env.logger_config import get_logger
logger = get_logger()

import os
import json
import torch
import mlflow
import numpy as np
import functools
import adapters.composition as ac
import copy

from omegaconf import DictConfig
from utils import mlops
from tqdm import tqdm
from multiprocessing import get_context
from transformers import Trainer, TrainingArguments, RobertaConfig, RobertaForSequenceClassification
from adapters import AdapterTrainer, RobertaAdapterModel
from utils import compute_metrics
from data_loaders.task_data_loader import TaskDataLoader

def handle_trial(model_variant, num_labels, dataset_name, training_args, adapter_config_name, dataset, trial, trainer_type, compute_metric, cfg):
    """
    Handles a single trial for Optuna optimization, setting up the model based on the specified
    parameters and executing the training process.

    Returns:
    - dict: Results of the trial including evaluation metrics.
    """

    training_args_class = TrainingArguments(**training_args)

    if trainer_type == 'adapter':
        assert adapter_config_name, "Adapter name is required for adapter training"
        if 'adapter_fusion' in trial.study.study_name:
            # Building a model that uses adapter fusion.
            # Note, this has been done for exploration purposes and may not be fully optimized.
            assert cfg.adapter_fusion, "Adapter fusion config is required for adapter fusion training"
            # Only support fusion of 2 adapters for now
            adapter1_path = cfg.adapter_fusion['adapter1_path']
            adapter2_path = cfg.adapter_fusion['adapter2_path']
            model = poc_build_adapter_fusion_model(model_variant, num_labels, dataset_name, adapter_config_name, adapter1_path, adapter2_path)
        else:
            model = build_adapter_model(model_variant, num_labels, dataset_name, adapter_config_name)
    elif trainer_type == 'model':
        model = build_classification_model(model_variant, num_labels)
    else:
        raise ValueError("Invalid training type. Supported types: model, adapter")
    
    # Run the trial with the created model
    return run_trial_for_seed(model, dataset, training_args_class, trial, trainer_type, compute_metric)


def get_dataset(dataset_name):
    """
    Loads and prepares the dataset for training a RoBERTa-based model variant.

    Parameters:
    - dataset_name (str): Name of the dataset to be loaded.

    Returns:
    - tuple: Contains the dataset object and the number of labels.
    """
    # ======================================================
    # Set-up and Load Data
    # ======================================================
    loader = TaskDataLoader(model_name="roberta-base",
                                    dataset_name=dataset_name,
                                    path=f"data/{dataset_name}/",
                                    checkpoint_path=f"data/{dataset_name}/processed_dataset.pt")

    dataset = loader.load_dataset(overwrite=False)
    num_labels = loader.num_labels

    return dataset, num_labels

def build_adapter_model(model_variant, num_labels, dataset_name, adapter_config_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ======================================================
    # Model & Adapter Config
    # ======================================================
    # Set up training for the Model and Adapter
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
    )
    
    model = RobertaAdapterModel.from_pretrained(model_variant, config=config)
    model.to(device)

    # Add adapter to model
    adapter_name = model_variant.split("/")[-1]+"_"+dataset_name+"_"+adapter_config_name
    model.add_adapter(adapter_name, config=adapter_config_name)
    model.add_classification_head(
        adapter_name,
        num_labels=num_labels
    )

    #  The train_adapter() method does two things:
    #     It freezes all weights of the pre-trained model, so only the adapter weights are updated during training.
    #     It activates the adapter and the prediction head such that both are used in every forward pass.
    # Activate the adapter
    model.train_adapter(adapter_name)

    return model


def poc_build_adapter_fusion_model(model_variant, num_labels, dataset_name, adapter_config_name, adapter1_path, adapter2_path):
    """
    For exploration, we test the fusion of two pre-selected adapters.
    There are some manual processing steps to ensure the correct adapter is loaded and named.

    Constructs a model by fusing two adapters. Sets up the model configuration, loads the pre-trained model,
    and fuses two adapters to explore transfer learning within adapters and potentially enhance performance on specific tasks .

    Returns:
    - RobertaAdapterModel: Configured model with fused adapters.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ======================================================
    # Model & Adapter Config
    # ======================================================
    # Set up training for the Model and Adapter
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
        problem_type="single_label_classification",
        hidden_dropout_prob=0.1,
    )
    
    model = RobertaAdapterModel.from_pretrained(model_variant, config=config)
    model.to(device)

    # Add adapter to model
    adapter_name = model_variant.split("/")[-1]+"_"+dataset_name+"_"+adapter_config_name+"_fusion"
    model.load_adapter(adapter1_path, load_as='ad1', with_head=False)
    model.load_adapter(adapter2_path, load_as='ad2', with_head=False)

    # Setup adapter fusion and add a classification head
    adapter_setup = [['ad1','ad2']]
    model.add_adapter_fusion(adapter_setup[0], "dynamic")
    model.train_adapter_fusion(adapter_setup)

    model.add_classification_head(
        adapter_name,
        num_labels=num_labels,
        overwrite_ok=True
    )

    model.train_adapter_fusion(ac.Fuse('ad1', 'ad2'))

    return model

def build_classification_model(model_variant, num_labels):
    """
    Creates a classification model using a specified variant of the Roberta model.
    This is used for our fine-tuning use cases for replicating the results of https://github.com/allenai/dont-stop-pretraining

    Returns:
    - RobertaForSequenceClassification: An configured classification model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
        problem_type="single_label_classification",
        hidden_dropout_prob=0.1,
    )

    model = RobertaForSequenceClassification.from_pretrained(model_variant, config=config)
    model.to(device)

    return model

def build_trainer(model, dataset, training_args_class, trainer_type, compute_metric):
    """
    Configures and returns a trainer for the model, depending on the trainer type.

    Parameters:
    - model (torch.nn.Module): The model to train.
    - dataset (dict): Dataset containing 'train' and 'dev' splits.
    - training_args_class (TrainingArguments): Configured training arguments.
    - trainer_type (str): Type of trainer to use ('model' or 'adapter').
    - compute_metric (str): Metric to compute (either 'macro_f1' or 'micro_f1').

    Returns:
    - Trainer or AdapterTrainer: Configured trainer for the model.
    """

    # Define the metric computation function based on user input
    if compute_metric == 'macro_f1':
        compute_metrics_fn = compute_metrics.macro_f1
    elif compute_metric == 'micro_f1':
        compute_metrics_fn = compute_metrics.micro_f1
    else:
        raise ValueError("Invalid metric. Supported metrics: macro_f1, micro_f1")

    if trainer_type == 'model':
        return Trainer(
            model=model,
            args=training_args_class,
            train_dataset=dataset['train'],
            eval_dataset=dataset['dev'],
            compute_metrics=compute_metrics_fn,
        )
    elif trainer_type == 'adapter':
        return AdapterTrainer(
            model=model,
            args=training_args_class,
            train_dataset=dataset['train'],
            eval_dataset=dataset['dev'],
            compute_metrics=compute_metrics_fn,
        )
    else:
        raise ValueError("Invalid training type. Supported types: model, adapter")

def run_trial_for_seed(model, dataset, training_args_class, trial, trainer_type='model', compute_metric='macro_f1'):
    """
    Executes a training trial for a given model and records results using Optuna/MLflow.
    Specifically, the trainer (if configured to log) should record logs related to the trainer_state that contains epoch train and eval metrics.
    On the same folder (as per training_args_class.output_dir), we write the mlflow_id.txt file that contains the mlflow run id. This can then be mapped to the mlflow UI/outputs for more granular info.
    In addition, for audit purposes, we write the training_args.json file that contains the training arguments used for the trial.

    Note, the input trial is an Optuna trial instance. As we run this in trials, we we make sure that logs do not overwrite each other by saving them in different directories.
    If seeds are configured, this should had been managed within the training_args_class definition of out_dir.

    Returns:
    - Evaluation results from the trial.
    """

    # logger.info(f"Starting process for: Seed={seed}: Trial {trial.number}")
    with mlflow.start_run(nested=True) as run:
        
        trainer = build_trainer(model, dataset, training_args_class, trainer_type, compute_metric)
        trainer.train()
        eval_results = trainer.evaluate(dataset["test"])

        # Save trainer state and log the mlrun
        trainer.save_state()

        # Save mlflow id in output_dir
        run_id = run.info.run_id
        with open(os.path.join(training_args_class.output_dir, "mlflow_id.txt"), "w") as f:
            f.write(run_id)

        # Save the TrainingArguments
        with open(os.path.join(training_args_class.output_dir, "training_args.json"), "w") as json_file:
            json.dump(training_args_class.to_dict(), json_file)

    output_dir = training_args_class.output_dir
    if 'adapter_v01_best' in output_dir:
        logger.warn("Best Adapter POC is enabled. Saving best adapter.")
        # POC -> Save best adapters
        # Note, this makes assumptions about model and base output dir
        # for instance, assumes the base output directory is 'training_output'
        out_fldr_base = "./adapters"
        if output_dir.startswith('./'):
            output_dir = output_dir.replace('./', '')

        out_fldr = f"{out_fldr_base}/{output_dir}"
        if not os.path.exists(out_fldr):
            os.makedirs(out_fldr)

        # assuming model only has one adapter attached
        adapter_name = list(model.adapters_config.adapters.keys())[0]

        logger.info(f"Saving adapter {adapter_name} to {out_fldr}")
        model.save_adapter(out_fldr, adapter_name)

    logger.info(f"{trial.study.study_name} >> Seed={training_args_class.seed}: Trial {trial.number} finished: Eval Loss {eval_results['eval_loss']}, Eval F1: {eval_results['eval_macro_f1']}")
    return eval_results

def run_study_experiments(cfg: DictConfig, model_variant, dataset_name, study_name, trainer_type='model', adapter_config_name=None, 
                          parallelism=False, job_sequence=1):
    """
    Optuna's role is broad. It looks at the performance of the model across different hyperparameter settings and different seeds 
    to find the set of hyperparameters that, on average, leads to the best score across runs. 
    Optuna does not concern itself with individual model checkpoints; it is focused on the hyperparameters that statistically result in the best performance.

    Parameters:
    - cfg (DictConfig): Configuration settings for the study.
    - model_variant (str): Model variant to use.
    - dataset_name (str): Dataset name.
    - study_name (str): Name of the study.
    - trainer_type (str): Type of trainer to use (model or adapter).
    - adapter_config_name (str, optional): Adapter configuration name, if applicable.
    - parallelism (bool): Experimental - Whether to run one trial in parallel for multiple seeds.
    - job_sequence (int): Job sequence number, used to handle multiple runs or re-runs.

    Handles configuration validation, data loading, and the execution of trials, logging results and handling errors.
    Optuna mantains a database, which we can interact through optuna-dashboard: https://github.com/optuna/optuna-dashboard
    e.g., install and run the below to view the dashboard:
        optuna-dashboard sqlite:///db.sqlite3
    """

    def objective(trial):
        n_seeds = cfg.optuna.n_seeds
        if len(cfg.seeds) < n_seeds:
            raise ValueError("Number of seeds in the configuration is less than the number of seeds required for Optuna trials. Check and align configs.")
        
        seeds = cfg.seeds[:n_seeds]
        results = []

        dataset, num_labels = get_dataset(dataset_name)
        training_args = mlops.build_training_arguments_for_trial(trial, cfg)

        pbar = tqdm(total=n_seeds, desc=f"Executing Trial {trial.number}", unit="seed")

        handle_trial_args = {
            'model_variant': model_variant,
            'num_labels': num_labels,
            'dataset_name': dataset_name,
            'training_args': training_args,
            'adapter_config_name': adapter_config_name,
            'dataset': dataset,
            'trial': trial,
            'trainer_type': trainer_type,
            'compute_metric': cfg.optuna.trainer_objective,
            'cfg': cfg # POC -> Added object to pass fusion config (i.e., the paths of adapters to fuse). This is to be refactored in future.
        }

        if parallelism:
            logger.warn("Experimental Feature: Running in parallel mode for multiple seeds simultaneously.")
            """
            Note, this appears to have some issues with the current setup.
            The results from different seeds does not seem to vary for unknown reasons.
            Therefore, for now this feature is disabled.
            Note, ensure all necessary resources and contexts are properly managed for multiprocessing.
            """
            # raise ValueError("Parallelism is currently disabled due to issues with the current setup.")
            ctx = get_context("spawn")
            with ctx.Pool() as pool:
                tasks = []
                
                for seed in seeds:
                    # Using partial to fix all arguments except seed
                    task_args = copy.deepcopy(handle_trial_args)
                    task_args['training_args'].update({'seed': seed})
                    task_args['training_args'].update({'output_dir': task_args['training_args']['output_dir'] + f'/seed_{seed}'})
                    task_func = functools.partial(handle_trial, **task_args)
                    
                    # Create and run task
                    task = pool.apply_async(task_func)
                    tasks.append(task)

                results = []
                for task in tqdm(tasks, total=len(seeds), desc="Processing Seeds"):
                    eval_results = task.get()
                    results.append(eval_results)
                    pbar.update(1)

        else:
            logger.info("Running in sequential mode for one seeds at a time.")
            for seed in seeds:
                task_args = copy.deepcopy(handle_trial_args)
                task_args['training_args'].update({'seed': seed})
                task_args['training_args'].update({'output_dir': task_args['training_args']['output_dir'] + f'/seed_{seed}'})
                eval_results = handle_trial(**task_args)
                results.append(eval_results)
                pbar.update(1)

        pbar.close()

        average_eval_loss = np.mean([x['eval_loss'] for x in results])
        average_f1 = np.mean([x['eval_macro_f1'] for x in results])
        std_dev_f1 = np.std([x['eval_macro_f1'] for x in results])

        # Log additional information into optuna
        trial.set_user_attr("average_eval_loss", average_eval_loss)
        trial.set_user_attr("average_f1", average_f1)
        trial.set_user_attr("std_dev_f1", std_dev_f1)

        logger.info(f"{trial.study.study_name} >> Trial {trial.number} finished: Avg Loss {average_eval_loss}, Avg F1: {average_f1}, Std Dev F1: {std_dev_f1}")

        if cfg.optuna.objective == 'f1':
            return average_f1
        elif cfg.optuna.objective == 'loss':
            return average_eval_loss
        else:
            raise ValueError("Invalid objective. Supported objectives: f1, loss")

    storage = "sqlite:///db.sqlite3"

    # If output dir or optuna dir already exists, ask user if want to continue (disabled)
    # Provides an option to stop process in case want to remediate
    study_exists = mlops.check_study_exists(study_name, storage)
    study_path = os.path.join(cfg.base_output_dir, study_name)
    if job_sequence == 1 and (os.path.exists(study_path) or study_exists):
        if study_exists:
            logger.warning(f"Study {study_name} already exists. Training will overwrite existing files.")
        if os.path.exists(study_path):
            logger.warning(f"Study output directory {study_path} already exists. Training will overwrite existing files.")
        # # Ask user if they want to continue
        # user_input = input("Do you want to continue? (yes/no): ").lower()

        # # Check user response
        # if user_input != "yes":
        #     print("Training process stopped.")
        #     import sys
        #     sys.exit()

    # Trials: During the optimization process, Optuna conducts multiple trials, 
    # each time evaluating the objective function with a different set of hyperparameters. 
    # These trials could be run sequentially or in parallel.
    # We could define an optimization algorithm for hyperparameter search. e.g., TPE, Random Search, Grid Search, etc.
    logger.info(f"Optuna study {study_name} objective is set to: {cfg.optuna.direction} the {cfg.optuna.objective} for the trainer {cfg.optuna.trainer_objective}.")
    study = mlops.create_or_load_study(study_name, storage, direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=cfg.optuna.n_trials)
