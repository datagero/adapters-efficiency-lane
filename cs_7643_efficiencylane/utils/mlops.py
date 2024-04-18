import optuna
from transformers import TrainingArguments

def create_or_load_study(study_name, storage, direction='minimize'):
    """
    Create a new Optuna study or load an existing one from storage.
    """
    try:
        # Try loading the study
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Study '{study_name}' loaded from storage.")
    except KeyError:
        # If the study does not exist, create a new one
        study = optuna.create_study(study_name=study_name, storage=storage, direction=direction)
        print(f"Study '{study_name}' created.")

    return study

def map_param_to_arg_names(param_name):
    """
    Map a hyperparameter name to one or more TrainingArguments fields.
    """
    # Mapping from parameter names to TrainingArguments fields
    mapping = {
        'learning_rate': ['learning_rate'],
        'batch_size': ['per_device_train_batch_size', 'per_device_eval_batch_size'],
    }
    return mapping.get(param_name, [param_name])

def build_training_arguments_for_trial(trial, cfg):
    """
    Build TrainingArguments for a given Optuna trial using the provided configuration.
    """
    # Dictionary to hold suggested hyperparameter values
    suggested_params = {}
    
    # Dynamically suggest values for all defined hyperparameters
    for param_name, param_config in cfg.optuna_search_space.items():
        if 'low' in param_config and 'high' in param_config:
            # Suggest a value using loguniform for continuous parameters
            suggested_value = trial.suggest_loguniform(param_name, param_config.low, param_config.high)
        elif 'values' in param_config:
            # Suggest a value using categorical for discrete parameters
            suggested_value = trial.suggest_categorical(param_name, param_config['values'])
        else:
            continue  # Skip if configuration is not recognized

        # Map parameter name to one or more argument names and update suggested_params
        arg_names = map_param_to_arg_names(param_name)
        for arg_name in arg_names:
            suggested_params[arg_name] = suggested_value

    # Setup output directory
    study_name = trial.study.study_name
    output_dir = f"{cfg.base_output_dir}/{study_name}/trial_{trial.number}/"

    # Initialize training arguments using both predefined and suggested parameters
    training_kwargs = cfg.training_args.copy()  # Ensures original is not modified
    training_kwargs.update({
        'output_dir': output_dir,
        **suggested_params
    })

    # Create and return TrainingArguments
    return TrainingArguments(**training_kwargs)

# Check final size of the adapter
# !ls -lh final_adapter

# Share Adapter to adapter hub
# model.push_adapter_to_hub(
#     "my-awesome-adapter",
#     "rotten_tomatoes",
#     adapterhub_tag="sentiment/rotten_tomatoes",
#     datasets_tag="rotten_tomatoes"
# )

