import logging
import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import classification_report
from experiments_utils import experiment_status

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


from sklearn.metrics import classification_report, balanced_accuracy_score, fbeta_score

def print_classification_reports(y_train, y_train_pred, y_test, y_test_pred, target_names=['ham', 'spam']):

    train_report = classification_report(y_train, y_train_pred, target_names=target_names, digits=3)
    test_report = classification_report(y_test, y_test_pred, target_names=target_names, digits=3)

    return train_report, test_report

def plot_roc_curve(pipeline, X_train, y_train, X_test, y_test):
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    pr_fig, ax = plt.subplots(figsize=(5, 5))      
    ax.plot(fpr_train, tpr_train, label=f'Train ROC (AUC = {auc_train:.3f})', color='blue', linewidth=2)
    ax.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {auc_test:.3f})', color='red', linestyle='--', linewidth=2)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random classifier')
    ax.title('ROC Curve', fontsize=16)
    ax.xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(alpha=0.3)
    return pr_fig
 
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_test, y_test_pred):
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    matrix_fig, ax = plt.subplots(figsize=(5, 5))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=test_conf_matrix, display_labels=['Ham', 'Spam'])
    cm_display.plot(cmap='Blues', ax=ax)
    plt.title("Confusion Matrix")
    return matrix_fig
    
from sklearn.metrics import precision_recall_curve as pr_curve

def plot_precision_recall_curve(pipeline, X_test, y_test):
    y_test_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, _ = pr_curve(y_test, y_test_pred_prob)
    pr_auc = auc(recall, precision)

    pr_fig, ax = plt.subplots(figsize=(5, 5))  
    ax.plot(recall, precision, color='b', label=f'PR AUC = {pr_auc:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    ax.grid(True)
    # plt.show()
    return pr_fig

    
    


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


    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)



    experiment_name, _, _ = experiment_status()
    mlflow.set_experiment(experiment_name)
    logging.info("Initializing experiment tracking...")

    try:
        with mlflow.start_run(run_name=run_name, log_system_metrics=True):
            logging.info("Calculating classification report...")
            print_classification_reports()
            logging.info("Plotting ROC_AUC and PR curves...")
            plot_roc_curve()
            plot_precision_recall_curve()
            logging.info("Plotting the Confusion Matrix...")
            plot_confusion_matrix()
            
            balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
            f0_5_score = fbeta_score(y_test, y_test_pred, beta=0.5)
            
            logging.info("Setting tags on MLFlow...")
            # Tags from YAML
            for key, value in experiment_tags.items():
                mlflow.set_tag(key, value)
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
