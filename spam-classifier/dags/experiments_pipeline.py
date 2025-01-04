from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import nlpaug.augmenter.word as naw
from typing import Callable
import logging
from ..components.preprocessing import preprocess
from ..components.data_ingestion import read_data
from time import timedelta

config_path = "./experiments_config.yaml"
experiment_config = load_yaml_config(config_path)

# DAG
default_args = {
    "owner": "Mldu",
    "depends_on_past": False,
    "email_on_failure": 'contactme@gmail.com',
    "email_on_retry": 'contactmeurgently@gmail.com',
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag =  DAG(
    "experiments_pipeline",
    default_args=default_args,
    description="Pipeline of experiments with MLFlow",
    schedule_interval=None,  
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

def experiment_no_nlpaug(df: pd.DataFrame) -> None:
    try:
        logging.info("Starting baseline model")
        df_ingested = read_data(df, )
        df_preprocessed = preprocess(df_ingested)
        logging.info("No NLPaug Experiment Results:")
        logging.info(df_preprocessed)
    except Exception as e:
        logging.error(f"Error during baseline_model experiment: {e}")

# Tarea para procesar usando SynonymAug
def experiment_synonymaug(df: pd.DataFrame) -> None:
    try:
        logger.info("Starting 'SynonymAug' experiment.")
        df_preprocessed = preprocess(df, use_nlpaug=True)
        aug = naw.SynonymAug(aug_p=0.1)  # Sinónimos con probabilidad del 10%
        df_preprocessed['Message'] = df_preprocessed['Message'].apply(aug.augment)
        # Aquí puedes guardar los resultados en un archivo o base de datos
        logger.info("SynonymAug Experiment Results:")
        logger.info(df_preprocessed)
    except Exception as e:
        logger.error(f"Error during 'SynonymAug' experiment: {e}")

# Tarea para procesar usando SpellingAug
def experiment_spellingaug(df: pd.DataFrame) -> None:
    try:
        logger.info("Starting 'SpellingAug' experiment.")
        df_preprocessed = preprocess(df, use_nlpaug=True)
        aug = naw.SpellingAug(aug_p=0.1)  # Aumento de errores de ortografía con probabilidad del 10%
        df_preprocessed['Message'] = df_preprocessed['Message'].apply(aug.augment)
        # Aquí puedes guardar los resultados en un archivo o base de datos
        logger.info("SpellingAug Experiment Results:")
        logger.info(df_preprocessed)
    except Exception as e:
        logger.error(f"Error during 'SpellingAug' experiment: {e}")