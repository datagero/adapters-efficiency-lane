"""
Here we define the Optuna objective used during our training processes
We support parallelism for different random seeds.
The results from the seeds get averaged and the average is used as the final result for the trial.
"""
from env.logger_config import get_logger
logger = get_logger()

import mlflow
import numpy as np

from omegaconf import DictConfig
from utils import mlops
from tqdm import tqdm
from multiprocessing import get_context
from transformers import Trainer
from adapters import AdapterTrainer
from utils import compute_metrics

def run_trial_for_seed(model, dataset, cfg, trial, seed):
    # logger.info(f"Starting process for: Seed={seed}: Trial {trial.number}")
    # with mlflow.start_run(nested=True):
    training_args = mlops.build_training_arguments_for_trial(trial, cfg, seed)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics.macro_f1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # Save trainer state
    trainer.save_state()

    logger.info(f"Seed={seed}: Trial {trial.number} finished: Eval Loss {eval_results['eval_loss']}, Eval F1: {eval_results['eval_macro_f1']}")
    return eval_results

def run_trial_for_seed_adapter(model, dataset, cfg, trial, seed):
    logger.info(f"Starting process for: Seed={seed}: Trial {trial.number}")
    with mlflow.start_run(nested=True):
        training_args = mlops.build_training_arguments_for_trial(trial, cfg, seed)
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            compute_metrics=compute_metrics.macro_f1,
        )

        trainer.train()
        eval_results = trainer.evaluate()

        logger.info(f"Seed={seed}: Trial {trial.number} finished: Eval Loss {eval_results['eval_loss']}, Eval F1: {eval_results['eval_macro_f1']}")
        return eval_results

def run_study_experiments(cfg: DictConfig, model, dataset, study_name, training_type='model', parallelism=False):

    """
    Optuna's role is broad. It looks at the performance of the model across different hyperparameter settings and different seeds 
    to find the set of hyperparameters that, on average, leads to the best score across runs. 
    Optuna does not concern itself with individual model checkpoints; it is focused on the hyperparameters that statistically result in the best performance.
    """

    def objective(trial):
        n_seeds = cfg.optuna.n_seeds
        if len(cfg.seeds) < n_seeds:
            raise ValueError("Number of seeds in the configuration is less than the number of seeds required for Optuna trials. Check and align configs.")
        
        seeds = cfg.seeds[:n_seeds]
        results = []

        pbar = tqdm(total=n_seeds, desc=f"Executing Trial {trial.number}", unit="seed")

        if parallelism:
            """
            Note, this appears to have some issues with the current setup.
            The results from different seeds does not seem to vary for unknown reasons.
            Therefore, for now this feature is dissabled.
            """
            raise ValueError("Parallelism is currently disabled due to issues with the current setup.")
            # ctx = get_context("spawn")
            # with ctx.Pool() as pool:
            #     # Pack the extra arguments for each seed
            #     if training_type == 'model':
            #         tasks = [pool.apply_async(run_trial_for_seed, (model, dataset, cfg, trial, seed)) for seed in seeds]
            #     elif training_type == 'adapter':
            #         tasks = [pool.apply_async(run_trial_for_seed_adapter, (model, dataset, cfg, trial, seed)) for seed in seeds]
            #     else:
            #         raise ValueError("Invalid training type. Supported types: model, adapter")

            #     for task in tqdm(tasks, total=len(seeds), desc="Processing Seeds"):
            #         eval_results = task.get()
            #         results.append(eval_results)  # Assuming eval_results is a list
            #         pbar.update(1)

        else:
            for seed in seeds:
                if training_type == 'model':
                    eval_results = run_trial_for_seed(model, dataset, cfg, trial, seed)
                elif training_type == 'adapter':
                    eval_results = run_trial_for_seed_adapter(model, dataset, cfg, trial, seed)
                else:
                    raise ValueError("Invalid training type. Supported types: model, adapter")

                results.append(eval_results)
                pbar.update(1)

        pbar.close()

        average_eval_loss = np.mean([ x['eval_loss'] for x in results])
        average_f1 = np.mean([x['eval_macro_f1'] for x in results])
        std_dev_f1 = np.std([x['eval_macro_f1'] for x in results])

        # Log additional information
        trial.set_user_attr("average_eval_loss", average_eval_loss)
        trial.set_user_attr("average_f1", average_f1)
        trial.set_user_attr("std_dev_f1", std_dev_f1)

        logger.info(f"Trial {trial.number} finished: Avg Loss {average_eval_loss}, Avg F1: {average_f1}, Std Dev F1: {std_dev_f1}")

        return -average_f1  # For f1 maximization (note the negative sign)

    # Trials: During the optimization process, Optuna conducts multiple trials, 
    # each time evaluating the objective function with a different set of hyperparameters. 
    # These trials could be run sequentially or in parallel.
    # We could define an optimization algorithm for hyperparameter search. e.g., TPE, Random Search, Grid Search, etc.
    storage = "sqlite:///db.sqlite3"
    study = mlops.create_or_load_study(study_name, storage, direction='minimize')
    study.optimize(objective, n_trials=cfg.optuna.n_trials)
