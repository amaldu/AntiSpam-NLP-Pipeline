import os
import logging
import mlflow
from .experiments_utils import (
    experiment_status, 
    print_classification_reports, 
    plot_roc_curve, 
    plot_confusion_matrix, 
    plot_precision_recall_curve,    

)
from sklearn.metrics import ( fbeta_score, balanced_accuracy_score )
import sys




########################### for testing 
from dotenv import load_dotenv
import os
import mlflow

load_dotenv()
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_default_region = os.getenv("AWS_DEFAULT_REGION")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))



sys.path.append("../utils")
sys.stdout.flush()

from .experiments_utils import experiment_status

experiment_name, _, _ = experiment_status()
mlflow.set_experiment(experiment_name)


import boto3
boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_default_region)
#######################







def log_experiment_with_mlflow(
    config_path=config_path,
    run_name,
    params,
    metrics,
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    yaml_path="config.yaml",
    vectorizer=None,
    augmentation_type=None,
    model_type=None,
    signature=None,
):
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name, experiment_tags, experiment_defaults = experiment_status()
    mlflow.set_experiment(experiment_name)
    logging.info("Initializing experiment tracking...")

    try:
        with mlflow.start_run(run_name=run_name, log_system_metrics=True):
            logging.info("Calculating classification report...")
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)
            train_report, test_report = print_classification_reports(y_train, y_train_pred, y_test, y_test_pred)

            logging.info("Plotting ROC_AUC and PR curves...")
            roc_curve_fig = plot_roc_curve(pipeline, X_train, y_train, X_test, y_test)
            pr_curve_fig = plot_precision_recall_curve(pipeline, X_test, y_test)

            logging.info("Plotting the Confusion Matrix...")
            confusion_matrix_fig = plot_confusion_matrix(y_test, y_test_pred)

            balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
            f0_5_score = fbeta_score(y_test, y_test_pred, beta=0.5)

            logging.info("Setting tags on MLFlow...")
            for key, value in experiment_tags.items():
                mlflow.set_tag(key, value)
            for key, value in experiment_defaults.items():
                mlflow.set_tag(key, value)
            mlflow.set_tag("vectorizer", vectorizer)
            mlflow.set_tag("augmentation_type", augmentation_type)
            mlflow.set_tag("model_type", model_type)

            logging.info("Logging parameters on MLFlow...")
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            logging.info("Logging metrics on MLFlow...")
            metrics["balanced_accuracy"] = balanced_accuracy
            metrics["f0_5_score"] = f0_5_score
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            logging.info("Logging figures on MLFlow...")
            mlflow.log_figure(roc_curve_fig, "roc_curve.png")
            mlflow.log_figure(pr_curve_fig, "precision_recall_curve.png")
            mlflow.log_figure(confusion_matrix_fig, "confusion_matrix.png")

            logging.info("Logging classification reports as artifacts...")
            with open("train_classification_report.txt", "w") as train_report_file:
                train_report_file.write(train_report)
            with open("test_classification_report.txt", "w") as test_report_file:
                test_report_file.write(test_report)

            mlflow.log_artifact("train_classification_report.txt")
            mlflow.log_artifact("test_classification_report.txt")

            logging.info("Logging pipeline on MLFlow...")
            mlflow.sklearn.log_model(pipeline, "pipeline", signature=signature)

    except Exception as e:
        logging.error(f"Error executing MLflow commands: {e}")

    logging.info("MLflow run terminated")
