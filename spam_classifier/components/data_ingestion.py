import pandas as pd
import logging
import sys

import os 
import yaml
import logging
import exception

def read_dataset(config_path: str) -> pd.DataFrame:
    logging.info(f"Loading config from {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        dataset_dir = os.path.abspath(config['dataset']['path'])
        dataset_file = config['dataset']['file']
        dataset_path = os.path.join(dataset_dir, dataset_file)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset '{dataset_path}' does not exist.")

        data = pd.read_csv(dataset_path)

        if data.empty:
            raise ValueError(f"The dataset '{dataset_path}' is empty or malformed.")

        logging.info(f"Dataset successfully loaded from {dataset_path} with shape: {data.shape}")
        return data

    except Exception as e:
        logging.error(f"Error reading the data: {e}")
        raise ValueError(f"An error occurred while reading the dataset: {e}")
