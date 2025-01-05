import pandas as pd
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
import os 
import yaml


def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config

    except FileNotFoundError:
        raise FileNotFoundError(f"The config file {config_path} was not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Error reading the YAML: {e}")

def read_dataset(config_path: str) -> pd.DataFrame:

    config = load_config(config_path)
    base_path = config['path']
    dataset_file = config['file']
    dataset_path = os.path.join(base_path, dataset_file)

    logging.info(f"Reading the dataset from {dataset_path}")
    
    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset '{dataset_path}' does not exist.")
        
        data = pd.read_csv(dataset_path)

        if data.empty:
            raise ValueError(f"The dataset '{dataset_path}' is empty or malformed.")
        
        return data

    except Exception as e:
        logging.error(f"Error reading the data: {e}")
        raise ValueError(f"An error occurred while reading the dataset '{dataset_path}': {e}")





