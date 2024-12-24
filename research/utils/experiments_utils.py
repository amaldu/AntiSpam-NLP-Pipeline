import logging
import random
import pandas as pd
import os
import time
from itertools import cycle
from nltk.corpus import wordnet
import yaml


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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



import os
import psutil

def log_system_resources():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    logging.info("Memory Usage: %s MB (Total: %s MB)", memory.used / (1024 * 1024), memory.total / (1024 * 1024))
    logging.info("CPU Usage: %s%%", cpu)



