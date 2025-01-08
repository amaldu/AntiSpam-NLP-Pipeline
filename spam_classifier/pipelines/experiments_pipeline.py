import logging
from sklearn.pipeline import Pipeline
from spam_classifier.components.data_ingestion import read_dataset
from spam_classifier.components.data_preprocessing import clean_types_duplicates, preprocess
from spam_classifier.components.split_data import train_test_val_split
from spam_classifier.components.vectorizers import BowFeatureExtractor, TfidfFeatureExtractor
from spam_classifier.components.models import MultinomialNaiveBayesClassifier, LogisticRegressionClassifier
import os
import mlflow
import mlflow.sklearn

from spam_classifier.utils.mlflow_utils import log_experiment_with_mlflow 


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
            ('model', MultinomialNaiveBayesClassifier())  
        ])
        
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        params = {
            'vectorizer': type(pipeline.named_steps['vectorizer']).__name__,
            'classifier': type(pipeline.named_steps['classifier']).__name__
        }
        metrics = {} 
        
        logging.info("Logging experiment with MLflow...")
        log_experiment_with_mlflow(
            run_name="Baseline Experiment v1",
            params=params,
            metrics=metrics,
            figures={},  
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            y_train_pred=y_train_pred,
            X_test=X_test,
            y_test=y_test,
            y_test_pred=y_test_pred
        )

    except Exception as e:
        logging.error(f"baseline_experiment_v1: {e}")

def main():
    y_train_pred, y_test_pred = baseline_experiment_v1()
    print("Train predictions:", y_train_pred)
    print("Test predictions:", y_test_pred)

if __name__ == "__main__":
    main()
