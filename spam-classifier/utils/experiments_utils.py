import logging
import os
import yaml
import psutil
import mlflow


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def experiment_status(config_file='experiments_config.yaml'):
    """
    Loads config from yaml file.
    
    Args:
    - config_file (str): file root to YAML.
    
    Returns:
    - tuple: (experiment_name, experiment_description, experiment_tags)
    """
    try:
        script_dir = os.path.dirname(__file__)  
        absolute_config_path = os.path.join(script_dir, config_file)
        
        logging.info(f"Loading configuration from: {absolute_config_path}")
        
        if not os.path.exists(absolute_config_path):
            logging.error(f"File not found: {absolute_config_path}")
            raise FileNotFoundError(f"The file {absolute_config_path} does not exist.")
        
        with open(absolute_config_path, 'r') as file:
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


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


from sklearn.metrics import classification_report, balanced_accuracy_score, fbeta_score

def print_classification_reports(y_train, y_train_pred, y_test, y_test_pred, target_names=['ham', 'spam']):
    """
    Generates and prints the classification reports for training and test data.

    Parameters:
        y_train (array-like): True labels for the training data.
        y_train_pred (array-like): Predicted labels for the training data.
        y_test (array-like): True labels for the test data.
        y_test_pred (array-like): Predicted labels for the test data.
        target_names (list, optional): Names of the target classes. Default is None.
    """
    train_report = classification_report(y_train, y_train_pred, target_names=target_names, digits=3)
    print("Classification Report (Train Data):")
    print(train_report)

    test_report = classification_report(y_test, y_test_pred, target_names=target_names, digits=3)
    print("Classification Report (Test Data):")
    print(test_report)





def plot_roc_curve(pipeline, X_train, y_train, X_test, y_test):
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr_train, tpr_train, label=f'Train ROC (AUC = {auc_train:.3f})', color='blue', linewidth=2)
    plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {auc_test:.3f})', color='red', linestyle='--', linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random classifier')
    plt.title('ROC Curve', fontsize=16)
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



    
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
    # Get predicted probabilities for class 1
    y_test_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate precision, recall, and thresholds
    precision, recall, _ = pr_curve(y_test, y_test_pred_prob)
    pr_auc = auc(recall, precision)

    # Plot Precision-Recall curve
    pr_fig, ax = plt.subplots(figsize=(5, 5))  
    ax.plot(recall, precision, color='b', label=f'PR AUC = {pr_auc:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    ax.grid(True)
    # plt.show()
    return pr_fig

    
    
from colorama import Fore


def print_and_highlight_diff(orig_text, new_texts):
    """ A simple diff viewer for augmented texts. """
    for orig_text, new_text in zip(orig_text, new_texts):
        orig_split = orig_text.split()
        print("-"*50) 
        print(f"Original: {len(orig_split)}\n{orig_text}")
        print(f"Augmented: {len(new_text.split())}")
        
        for i, word in enumerate(new_text.split()):
            if i < len(orig_split) and word == orig_split[i]:
                print(word, end=" ")
            else:
                print(Fore.RED + word + Fore.RESET, end=" ")
                
        print()