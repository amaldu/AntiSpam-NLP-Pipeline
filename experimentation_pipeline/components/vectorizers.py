from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import logging


def apply_vectorizer(config_file: str, X: pd.Series) -> pd.Series:
    try: 
        logging.info("Logging the vectorizer typeand params from config file")
        vectorizer_type = config_file['vectorizer']['type']
        vectorizer_params = config_file['vectorizer']['params']
        logging.info(f"Vectorizer type is {vectorizer_type}")
        logging.info (f"Vectorizer params are {vectorizer_params}")
        
        if vectorizer_type == "bow":
            return CountVectorizer(**vectorizer_params)
        elif vectorizer_type == "tfidf":
            return TfidfVectorizer(**vectorizer_params)
        else:
            logging.error("Unknown vectorizer type or invalid format")
            raise
    except ValueError as e:
        logging.error(f"Unsupported vectorizer type: {vectorizer_type}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        raise
    

