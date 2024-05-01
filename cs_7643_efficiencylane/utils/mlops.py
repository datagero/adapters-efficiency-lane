"""
Supports operations related to managing and executing ML training with the transformers package and Optuna hyperparameter search framework.
It includes functions to check for the existence of Optuna studies, create or load studies, handle retries, map hyperparameters to model argument names, suggest parameter values,
build output directories, and construct training arguments for trials.
"""

from env.logger_config import get_logger
logger = get_logger()

import time
import optuna
from retry import retry

def check_study_exists(study_name, storage="sqlite:///db.sqlite3"):
    """
    Checks if an Optuna study exists in the provided storage.

    Returns:
    - bool: True if the study exists, False otherwise.
    """
    return study_name in optuna.study.get_all_study_names(storage)

@retry(KeyError, delay=3, tries=2)
def load_study_with_retry(study_name, storage):
    """
    Retry loading the study with a delay of 3 seconds, up to 2 attempts.
    """
    return optuna.load_study(study_name=study_name, storage=storage)

def create_or_load_study(study_name, storage, direction='minimize'):
    """
    Create a new Optuna study or load an existing one from storage.
    Make sure that the optimization direction matches the metric being optimized, as per your complete optuna configuration.
    """
    # try:
    #     # Try loading the study
    #     study = load_study_with_retry(study_name, storage)
    #     print(f"Study '{study_name}' loaded from storage.")
    # except KeyError:
    # If the study does not exist, create a new one
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name=study_name, storage=storage, direction=direction, pruner=pruner, load_if_exists=True)
    print(f"Study '{study_name}' created or reloaded.")

    return study

def map_param_to_arg_names(param_name):
    """
    Map a hyperparameter name to one or more TrainingArguments fields.
    This is mostly for cases where the hyperparameter values are searched over by Optuna, 
    and we need the same value for multiple arguments (e.g., train/eval batch size).
    """
    # Mapping from parameter names to TrainingArguments fields
    mapping = {
        'learning_rate': ['learning_rate'],
        'batch_size': ['per_device_train_batch_size', 'per_device_eval_batch_size'],
    }
    return mapping.get(param_name, [param_name])

def build_study_output_dir(base_output_dir, study_name, trial_number, seed):
    """
    Build the output directory for a given Optuna study using the provided configuration.
    Note, seed may be None from upstream processess, in which case we only run one seed per optuna trial.
    Otherwise, an Optuna trial is conformed from multiple seeds, which results should get averaged out by downstream processes.
    """
    if seed:
        return f"{base_output_dir}/{study_name}/trial_{trial_number}/seed_{seed}/"
    else:
        return f"{base_output_dir}/{study_name}/trial_{trial_number}/"

def get_optuna_suggested_params(optuna_search_space, trial):
    """
    Get suggested hyperparameter values for a given Optuna trial using the provided configuration.

    Returns:
    - dict: Dictionary of suggested hyperparameters mapped to their respective argument names.
    """
    # Dictionary to hold suggested hyperparameter values
    suggested_params = {}
    
    # Dynamically suggest values for all defined hyperparameters
    for param_name, param_config in optuna_search_space.items():
        
        if 'low' in param_config and 'high' in param_config:
            # Get data type of the parameter
            param_type = param_config.type
            if param_type == 'int':
                # Suggest a value using quniform for integer parameters
                suggested_value = trial.suggest_int(param_name, param_config.low, param_config.high)
            elif param_type == 'float':
                suggested_value = trial.suggest_float(param_name, param_config.low, param_config.high)
            elif param_type == 'loguniform':
                # Suggest a value using loguniform for continuous parameters
                suggested_value = trial.suggest_loguniform(param_name, param_config.low, param_config.high)
            elif param_type == 'uniform':
                # Suggest a value using uniform for continuous parameters
                suggested_value = trial.suggest_uniform(param_name, param_config.low, param_config.high)
            else:
                raise(ValueError(f"Unsupported parameter type: {param_type}"))
        elif 'values' in param_config:
            # Suggest a value using categorical for discrete parameters
            suggested_value = trial.suggest_categorical(param_name, param_config['values'])
        else:
            continue  # Skip if configuration is not recognized

        # Map parameter name to one or more argument names and update suggested_params
        arg_names = map_param_to_arg_names(param_name)
        for arg_name in arg_names:
            suggested_params[arg_name] = suggested_value

    logger.info(f"Suggested hyperparameters for trial {trial.number}:\n{suggested_params}")
    return suggested_params

def build_training_arguments_for_trial(trial, cfg, seed=None):
    """
    Build arguments for a TrainingArguments for a given Optuna trial using the provided configuration.
    In specific, this function suggests hyperparameter values for the trial and initializes the training arguments.
    Also, build the output directory for the trial using the study name, trial number, and seed.
    """
    if 'optuna_search_space' in cfg:
        suggested_params = get_optuna_suggested_params(cfg.optuna_search_space, trial)
    else:
        # No need to suggest parameters if no search space is defined
        suggested_params = {}

    # Initialize training arguments using both predefined and suggested parameters
    training_kwargs = cfg.training_args.copy()  # Ensures original is not modified
    training_kwargs.update({
        'output_dir': build_study_output_dir(cfg.base_output_dir, trial.study.study_name, trial.number, seed),
        **suggested_params
    })

    if seed:
        training_kwargs['seed'] = seed

    # arguments for TrainingArguments
    return training_kwargs
