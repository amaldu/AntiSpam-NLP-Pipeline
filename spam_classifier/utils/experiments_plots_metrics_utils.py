import logging
import os
import yaml
import psutil
import mlflow


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def experiment_status(config_file='experiments_config.yaml'):
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

from sklearn.metrics import classification_report

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
    