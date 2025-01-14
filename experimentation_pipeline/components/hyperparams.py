from typing import Dict, Tuple, Callable, Union
import os

import mlflow
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from src.preprocessing import get_preprocessing_pipeline
from src.logger import get_console_logger

logger = get_console_logger()

# Function to sample hyperparameters for each model
def sample_hyperparams(
    model_fn: Callable,
    trial: optuna.trial.Trial,
) -> Dict[str, Union[str, int, float]]:
    if model_fn == LogisticRegression:
        return {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
        }
    elif model_fn == MultinomialNB:
        return {
            'alpha': trial.suggest_float('alpha', 0.01, 1.0, log=True)
        }
    elif model_fn == SVC:
        return {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
    else:
        raise NotImplementedError('Model not supported')

# Function to find the best hyperparameters
def find_best_hyperparams(
    model_fn: Callable,
    hyperparam_trials: int,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[Dict, Dict]:
    """Hyperparameter tuning for classification models"""
    assert model_fn in {LogisticRegression, MultinomialNB, SVC}

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Error function to minimize using hyperparameter tuning.
        """
        # Sample hyperparameters
        preprocessing_hyperparams = {
            'pp_rsi_window': trial.suggest_int('pp_rsi_window', 5, 20),
        }
        model_hyperparams = sample_hyperparams(model_fn, trial)
        
        # Evaluate the model using TimeSeriesSplit cross-validation
        tss = TimeSeriesSplit(n_splits=3)
        scores = []
        logger.info(f'{trial.number=}')
        for split_number, (train_index, val_index) in enumerate(tss.split(X)):

            # Split data for training and validation
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            logger.info(f'{split_number=}')
            logger.info(f'{len(X_train)=}')
            logger.info(f'{len(X_val)=}')

            # Train the model
            pipeline = make_pipeline(
                get_preprocessing_pipeline(**preprocessing_hyperparams),
                model_fn(**model_hyperparams)
            )
            pipeline.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = pipeline.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            scores.append(acc)
            
            logger.info(f'{acc=}')

        # Return the mean accuracy
        return -np.array(scores).mean()  # Negative because Optuna minimizes

    logger.info('Starting hyper-parameter search...')

    # Initialize MLflow experiment
    mlflow.set_experiment("classification_hyperparameter_tuning")
    with mlflow.start_run():
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=hyperparam_trials)

        # Get the best hyperparameters and their values
        best_params = study.best_params
        best_value = -study.best_value  # Convert back to positive

        # Log parameters and metrics to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("Best_Accuracy", best_value)

        # Split best_params into preprocessing and model hyper-parameters
        best_preprocessing_hyperparams = {
            key: value for key, value in best_params.items() 
            if key.startswith('pp_')
        }
        best_model_hyperparams = {
            key: value for key, value in best_params.items() 
            if not key.startswith('pp_')
        }

        logger.info("Best Parameters:")
        for key, value in best_params.items():
            logger.info(f"{key}: {value}")
        logger.info(f"Best Accuracy: {best_value}")

    return best_preprocessing_hyperparams, best_model_hyperparams
