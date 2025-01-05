from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import nlpaug.augmenter.word as naw
from typing import Callable
import logging
from ..components.preprocessing import clean_types_duplicates, preprocess
from ..components.split_data import train_test_val_split
from ..components.data_ingestion import read_data
from time import timedelta
import yaml
import os

def load_config(config_file: str = 'experiments_config.yml') -> dict:
    try:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' does not exist.")
        
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuration loaded successfully from {config_file}")
            return config
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise ValueError(f"Error parsing the configuration file '{config_file}': {e}")
    except Exception as e:
        logging.error(f"Unexpected error while loading configuration: {e}")
        raise




# # DAG
# default_args = {
#     "owner": "Mldu",
#     "depends_on_past": False,
#     "email_on_failure": 'contactme!@gmail.com',
#     "email_on_retry": 'contactmeurgently!@gmail.com',
#     "retries": 2,
#     "retry_delay": timedelta(minutes=5),
# }

# dag =  DAG(
#     "experiments_pipeline",
#     default_args=default_args,
#     description="Pipeline of experiments with MLFlow",
#     schedule_interval=None,  
#     start_date=datetime(2025, 1, 1),
#     catchup=False,
#)
def experiment_1() -> None:
    try:
        logging.info("Starting 'No NLPaug' experiment.")
        config = load_config()
        base_path = config['dataset']['path']
        
        df_ingested = read_data(base_path)
        df_cleaned = clean_types_duplicates(df_ingested)
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(df_cleaned)
        df_preprocessed = preprocess(df_ingested)

        logging.info("No NLPaug Experiment Results:")
        logging.info(df_preprocessed)
    except Exception as e:
        logging.error(f"Error during 'No NLPaug' experiment: {e}")

