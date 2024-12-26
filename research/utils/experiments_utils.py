import logging
import os
import yaml
import psutil


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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



