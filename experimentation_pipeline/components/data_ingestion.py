import pandas as pd
import logging
import sys

import os 
import yaml
import logging
import exception

def read_dataset(base_config: str) -> pd.DataFrame:
    logging.info(f"Loading config from {base_config}")
    
    dataset_path = base_config['dataset']['path']
    dataset_file = base_config['dataset']['file']
    full_path = os.path.join(dataset_path, dataset_file)
    try:
        data = pd.read_csv(full_path)
        logging.info(f"Dataset loaded successfully from {full_path}.")
        return data
    except FileNotFoundError as e:
        logging.error(f"Dataset file not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        raise