import pandas as pd
import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
from typing import Dict, Union, Optional, Callable
import os
import click
import yaml

from experimentation_pipeline.utils.experiments_utils import experiment_status, parse_config
from experimentation_pipeline.components.data_ingestion import read_dataset


config_path = os.path.join(os.path.dirname(__file__), '..', 'experiment_configs', 'base_config.yaml')
# @click.command()
# @click.argument("config_file", type=click.Path(exists=True, dir_okay=False), default=os.path.join("experimentation_pipeline", "experiment_configs", "config_file.yaml"))
def base_model_pipeline():
    """
    Train a simple model using the configuration file provided.

    CONFIG_FILE: Path to the YAML configuration file.
    """
    logging.info("Loading basic configurations from base_config.yaml...")
    try:
        with open(config_path, "r") as f:
            basic_config = yaml.safe_load(f)
        logging.info("Basic configurations loaded successfully.")
        
        data = read_dataset(basic_config)

    except Exception as e:
        logging.error(f"Error loading the dataset: {e}")
    return data
    
    
    
    
    
    

    # mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    # mlflow.set_tracking_uri(mlflow_tracking_uri)
    # experiment_name, experiment_description, experiment_tags = experiment_status(config)
    # client = MlflowClient()
    
    # try:
    #     client.create_experiment(
    #         name=experiment_name,
    #         tags={"mlflow.note.content": experiment_description, **experiment_tags}
    #     )
    # except Exception as e:
    #     logging.info(f"Experiment '{experiment_name}' already exists or could not be created: {e}")

    # logging.info(f"Initializing experiment '{experiment_name}'...")
        

    # with mlflow.start_run():
    #     mlflow.set_tags(experiment_tags)
        
    # except Exception as e:
    #     logging.error(f"baseline_experiment_v1: {e}")


if __name__ == '__main__':

    base_model_pipeline()

