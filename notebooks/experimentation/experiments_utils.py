import yaml
import os

def experiment_status(config_file='experiments_config.yaml'):
    """
    Loads config from yaml file
    
    Args:
    - config_file (str): file root to YAML.
    
    Returns:
    - tuple: (experiment_name, experiment_description, experiment_tags)
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The file {config_file} does not exist.")
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    experiment_name = config['experiment_name']
    experiment_description = config['experiment_description']
    experiment_tags = config['experiment_tags']
    
    return experiment_name, experiment_description, experiment_tags
