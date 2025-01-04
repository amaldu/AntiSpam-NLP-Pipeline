import logging
import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import classification_report


def load_config(yaml_path):
    """Loads the configuration from a YAML file."""
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def log_experiment_with_mlflow(
    run_name,
    params,
    metrics,
    figures,
    pipeline,
    yaml_path="config.yaml",
    vectorizer=None,
    augmentation_type=None,
    model_type=None,
    signature=None,
):
    """
    Logs an experiment to MLflow.

    Args:
        run_name (str): Name of the MLflow run.
        params (dict): Parameters to log.
        metrics (dict): Metrics to log.
        figures (dict): Figures to log (name as key, figure object as value).
        pipeline (object): Trained pipeline to log.
        yaml_path (str): Path to the YAML configuration file.
        vectorizer (str, optional): Vectorizer type (e.g., CountVectorizer).
        augmentation_type (str, optional): Type of data augmentation used.
        model_type (str, optional): Type of model used (e.g., Multinomial Naive Bayes).
        signature (mlflow.models.signature.ModelSignature, optional): Model signature.
    """
    # Load defaults from YAML
    config = load_config(yaml_path)
    experiment_defaults = config.get("experiment_defaults", {})

    logging.info("Initializing experiment...")

    try:
        with mlflow.start_run(run_name=run_name, log_system_metrics=True):
            logging.info("Setting tags on MLFlow...")
            # Tags from YAML
            for key, value in experiment_defaults.items():
                mlflow.set_tag(key, value)

            # Additional dynamic tags
            mlflow.set_tag("vectorizer", vectorizer)
            mlflow.set_tag("augmentation_type", augmentation_type)
            mlflow.set_tag("model_type", model_type)

            logging.info("Logging parameters on MLFlow...")
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            logging.info("Logging metrics on MLFlow...")
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            logging.info("Logging figures on MLFlow...")
            for figure_name, figure_obj in figures.items():
                mlflow.log_figure(figure_obj, figure_name)

            logging.info("Logging pipeline on MLFlow...")
            mlflow.sklearn.log_model(pipeline, "pipeline", signature=signature)

    except Exception as e:
        logging.error(f"Error executing MLflow commands: {e}")

    logging.info("MLflow run terminated")
