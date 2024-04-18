import yaml

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.SafeLoader)
    return yaml_data

def load_training_args_from_yaml(yaml_path):
    yaml_data = load_yaml(yaml_path)
    training_args = yaml_data['training_args']

    # Load the below as float
    if 'learning_rate' in training_args and training_args['learning_rate'] is not None:
        training_args['learning_rate'] = float(training_args['learning_rate'])

    if 'adam_epsilon' in training_args and training_args['adam_epsilon'] is not None:
        training_args['adam_epsilon'] = float(training_args['adam_epsilon'])

    return training_args
