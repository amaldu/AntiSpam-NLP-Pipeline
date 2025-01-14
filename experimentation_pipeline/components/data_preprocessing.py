import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import Any, Tuple
import os

from experimentation_pipeline.components.split_data import train_test_val_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from experimentation_pipeline.components.data_ingestion import read_dataset
import os
import yaml
import logging
import click

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()


def format_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    try:
        logging.info("Starting data cleaning: converting 'label' to int...")
        df['label'] = df['label'].map({"ham": 0, "spam": 1}).astype(int)
        
        logging.info("'label' column successfully converted to integer.")
        
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        final_count = len(df)
        logging.info(f"Duplicates removed: {initial_count - final_count} rows dropped.")
        
        logging.info (f"Reducing the dataframe into the columns of interest")
        df = df[["email", "label"]]
        
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(df)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    except KeyError as e:
        logging.error(f"KeyError: Missing expected column in DataFrame - {e}")
        raise
    except ValueError as e:
        logging.error(f"ValueError: Issue with data conversion - {e}")
        raise    
    except Exception as e:
        logging.error(f"Unexpected error during data cleaning: {e}")
        raise
    
    
    
    

class CustomTextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config_file: dict) -> None:
        self.config_file = config_file
        self.vectorizer = None

    def fit(self, X: pd.Series, y: pd.Series = None) -> 'CustomTextProcessor':

        logging.info("Fitting the CustomTextProcessor...")
        vectorizer_type = self.config_file['vectorizer']['type']
        vectorizer_params = self.config_file['vectorizer']['params']

        if vectorizer_type == "bow":
            self.vectorizer = CountVectorizer(**vectorizer_params)
        elif vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(**vectorizer_params)
        else:
            logging.error("Unsupported vectorizer type: %s", vectorizer_type)
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")

        # Fit the vectorizer
        self.vectorizer.fit(X)
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:

        logging.info("Transforming text data...")

        X_tokenized = tokenizer(self.config_file, X)
        X_no_stopwords = X_tokenized['tokens'].apply(remove_stopwords)
        X_lemmatized = X_no_stopwords.apply(lemmatize_text)
        transformed_data = self.vectorizer.transform(X_lemmatized)
        return transformed_data


def tokenizer(config_file: str, X: pd.Series) -> pd.Series:
    tokenizer_type = config_file['tokenizer']['type']
    
    if tokenizer_type == "nltk":
        logging.info(f"Tokenizing with nltk tokenizer")
        X_tokenized = X.apply(word_tokenize)  # Tokenize each entry in the series using nltk
        
    elif tokenizer_type == "spacy":
        logging.info(f"Tokenizing with spacy tokenizer")
        X_tokenized = X.apply(lambda x: [token.text for token in nlp(x)])  # Tokenize using spacy
    
    else:
        logging.error(f"Unsupported tokenizer type: {tokenizer_type}")
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    
    return X_tokenized  


def remove_stopwords(tokens: pd.Series) -> pd.Series:
    """Remove stopwords from a pandas Series of tokenized words."""
    return tokens.apply(lambda word_list: [word for word in word_list if word.lower() not in stop_words])

def lemmatize_text(tokens: pd.Series) -> pd.Series:
    """Lemmatize a pandas Series of tokenized words."""
    return tokens.apply(lambda word_list: ' '.join([lemmatizer.lemmatize(word) for word in word_list]))



def get_preprocessing_pipeline(config_file: dict) -> Pipeline:
    return make_pipeline(
        FunctionTransformer(format_data),
        CustomTextProcessor(config_file)
    )



# @click.command()
# @click.argument("config_file", type=click.Path(exists=True))
# def main(config_file):
#     # Cargar configuraciones
#     base_config_path = os.path.join(os.path.dirname(__file__), '..', 'experiment_configs', 'base_config.yaml')
#     with open(base_config_path, "r") as f:
#         basic_config = yaml.safe_load(f)
#         logging.info("Basic configurations loaded successfully.")
        

#     with open(config_file, 'r') as file:
#         config = yaml.safe_load(file)
#     logging.info("Experiment configuration loaded successfully.")
#     print(config)

#     # Leer dataset
#     data = read_dataset(basic_config)
#     # Pipeline
#     pipeline = get_preprocessing_pipeline(config)
#     transformed_data = pipeline.fit_transform(data)

#     # Salida
#     print("Pipeline output:")
#     print(transformed_data)


# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     main()