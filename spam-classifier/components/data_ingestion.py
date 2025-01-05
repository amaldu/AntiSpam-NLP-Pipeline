import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
import os 


def read_data(base_path: str, dataset_file: str) -> pd.DataFrame:
    dataset_path = os.path.join(base_path, dataset_file)
    logging.info(f"Reading the dataset from {dataset_path}")
    
    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset '{dataset_path}' does not exist.")
        
        data = pd.read_csv(dataset_path)

        if data.empty:
            raise ValueError(f"The dataset '{dataset_path}' is empty or malformed.")
        
        return data

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"The dataset '{dataset_path}' is empty.")
        raise ValueError(f"The dataset '{dataset_path}' is empty and cannot be processed.")
    except pd.errors.ParserError:
        logging.error(f"Error parsing the dataset '{dataset_path}'.")
        raise ValueError(f"There was an error parsing the dataset '{dataset_path}'.")
    except Exception as e:
        logging.error(f"Error reading the data: {e}")
        raise ValueError(f"An error occurred while reading the dataset '{dataset_path}': {e}")




