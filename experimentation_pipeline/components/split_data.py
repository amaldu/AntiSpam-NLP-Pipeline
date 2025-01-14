import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple


def train_test_val_split(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    try:
        logging.info("Starting train-test-validation split...")

        if 'email' not in df.columns or 'label' not in df.columns:
            raise KeyError("The DataFrame must contain 'email' and 'label' columns.")

        X = df['email']
        y = df['label']  

        logging.debug("Splitting data into train and temporary sets...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

        logging.debug("Splitting temporary set into validation and test sets...")
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        logging.info("Train-test-validation split completed successfully.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    except KeyError as e:
        logging.error(f"KeyError: {e}")
        raise ValueError(f"Missing expected column in DataFrame: {e}")

    except Exception as e:
        logging.error(f"Error during train-test-validation split: {e}")
        raise ValueError(f"An error occurred during the train-test-validation split: {e}")

        
