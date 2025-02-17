import yaml
from box import Box
from pathlib import Path
import numpy as np
import os
import pickle
import logging

def load_yaml(path: Path):
    """
    Loads a YAML file and returns its contents as a Box object.

    Args:
        path (Path): The file path to the YAML file.

    Returns:
        Box: The parsed YAML content as a Box object.
    """
    try:
        with open(path, "r") as file:
            return Box(yaml.safe_load(file))
    except Exception as e:
        logging.error(f"Error loading YAML file {path}: {e}")
        raise
    
def save_numpy_array(data: np.array, path: Path):
    """
    Saves a NumPy array to a specified file path.

    Args:
        data (np.array): The NumPy array to be saved.
        path (Path): The file path where the array should be saved.
    """
    try:
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        np.save(path, data)
    except Exception as e:
        logging.error(f"Error saving NumPy array to {path}: {e}")
        raise

def load_numpy_array(path: Path):
    """
    Loads a NumPy array from a specified file path.

    Args:
        path (Path): The file path.

    Returns:
        np.array: The loaded NumPy array.
    """
    try:
        with open(path, 'rb') as file:
            return np.load(file)
    except Exception as e:
        logging.error(f"Error loading NumPy array from {path}: {e}")
        raise

def save_object(model, path: Path):
    """
    Saves a Python object (e.g., a machine learning model) to a specified file path using Pickle.

    Args:
        model (object): The object to be saved.
        path (Path): The file path where the object should be saved.
    """
    try:
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        logging.error(f"Error saving object to {path}: {e}")
        raise

def load_object(path: Path):
    """
    Loads a Python object (e.g., a machine learning model) from a specified file path using Pickle.

    Args:
        path (Path): The file path from which the object should be loaded.

    Returns:
        object: The loaded Python object.
    """
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Error loading object from {path}: {e}")
        raise
