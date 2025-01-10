import pandas as pd
import re
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import numpy as np
from ..utils.textaug_techniques import TextAugmentation

stop_words = set(stopwords.words('english'))



def clean_types_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Starting data cleaning: mapping 'Category' and converting 'Message' to string...")
        df['Category'] = df['Category'].map({"ham": 0, "spam": 1}).astype(int)
        df['Message'] = df['Message'].astype(str)
        logging.info("'Category' and 'Message' columns successfully converted to string.")
        
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        final_count = len(df)
        logging.info(f"Duplicates removed: {initial_count - final_count} rows dropped.")
        
        return df
    
    except KeyError as e:
        logging.error(f"KeyError: Missing expected column in DataFrame - {e}")
        raise
    
    except ValueError as e:
        logging.error(f"ValueError: Issue with data conversion - {e}")
        raise
    
    except Exception as e:
        logging.error(f"Unexpected error during data cleaning: {e}")
        raise



def clean_text(text: str) -> str:
    special_replacements = {
        r"£": "pound",
        r"\$": "dollar",
        r"\€": "euro",
        r"%": "percentage", 
        r"ì": "i",
        r"ü": "you",
    }
    
    emoticon_pattern = re.compile(r"""
    [:;=Xx]           
    [-~]?             
    [\)\]\(\[dDpP/]   
    """, re.VERBOSE)
    text = text.lower()
    
    for pattern, replacement in special_replacements.items():
        text = re.sub(pattern, replacement, text)
    
    text = re.sub(emoticon_pattern, 'emoji', text)
    text = re.sub('<[^<>]+>', ' ', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub('[0-9]+', 'number', text)
    text = re.sub('[^\s]+@[^\s]+', 'emailaddr', text)
    
    text = text.translate(str.maketrans('', '', punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(tokens: list) -> list:
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def lemmatize_text(tokens: list) -> str:
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def preprocess(X_train: pd.Series, X_val: pd.Series, X_test: pd.Series, 
               y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> tuple:
    try:
        logging.info("Starting text preprocessing...")
        
        processed_texts = []

        for text in X_train:
            logging.debug("Cleaning text...")
            cleaned_text = clean_text(str(text))
            
            logging.debug("Tokenizing text...")
            tokens = word_tokenize(cleaned_text)
            
            logging.debug("Removing stopwords...")
            text_without_stopwords = remove_stopwords(tokens)
            
            logging.debug("Lemmatizing text...")
            preprocessed_text = lemmatize_text(text_without_stopwords)
            
            processed_texts.append(preprocessed_text)

        X_train_processed = np.array(processed_texts)
        
        X_train = X_train_processed
        X_val = X_val.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
        y_test = y_test.to_numpy()

        logging.info("Text preprocessing completed successfully.")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise ValueError(f"An error occurred during preprocessing: {e}")