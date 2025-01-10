import logging
import os
import yaml
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def experiment_status(config_path: str):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        experiment_name = config.get('experiment_name')
        experiment_description = config.get('experiment_description')
        experiment_tags = config.get('experiment_tags')
        
        logging.info("Configuration successfully loaded.")
        return experiment_name, experiment_description, experiment_tags
    
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        raise
    
    except yaml.YAMLError as e:
        logging.error(f"YAML parsing error: {e}")
        raise
    
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise


def log_system_resources():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    logging.info("Memory Usage: %s MB (Total: %s MB)", memory.used / (1024 * 1024), memory.total / (1024 * 1024))
    logging.info("CPU Usage: %s%%", cpu)

from sklearn.metrics import classification_report

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
