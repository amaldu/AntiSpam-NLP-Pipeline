import logging
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from spam_classifier.components.data_ingestion import read_dataset
from spam_classifier.components.data_preprocessing import clean_types_duplicates, preprocess
from spam_classifier.components.split_data import train_test_val_split
from spam_classifier.components.vectorizers import BowFeatureExtractor, TfidfFeatureExtractor
from spam_classifier.components.models import MultinomialNaiveBayesClassifier, LogisticRegressionClassifier
import os

logging.basicConfig(filename='logs/app.log', level=logging.INFO)
def baseline_experiment_v1():
    config_path = os.path.join(os.path.dirname(__file__), "experiments_config.yaml")
    try:
        logging.info("Loading dataset...")
        data = read_dataset(config_path)
        
        logging.info("Cleaning types and duplicates...")
        df_cleaned = clean_types_duplicates(data)
        
        logging.info("Splitting into train, test and validation sets...")
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(df_cleaned)
        
        logging.info("Preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test  = preprocess(X_train, X_val, X_test, y_train, y_val, y_test)
        
        logging.info("Applying pipeline...")
        pipeline = Pipeline([
            ('vectorizer', BowFeatureExtractor()), 
            ('classifier', MultinomialNaiveBayesClassifier())  
        ])
        
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        return y_train_pred, y_test_pred

    except Exception as e:
        logging.error(f"baseline_experiment_v1: {e}")

def main():
    y_train_pred, y_test_pred = baseline_experiment_v1()
    print("Train predictions:", y_train_pred)
    print("Test predictions:", y_test_pred)

if __name__ == "__main__":
    main()
