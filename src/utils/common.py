import yaml
from box import Box
from pathlib import Path
import numpy as np
import os
import pickle


def load_yaml(path: Path):
    # Load YAML file
    with open(path, "r") as file:
        return Box(yaml.safe_load(file))
    

def save_numpy_array(data: np.array, path: Path):
    """
    Saves a NumPy array to a specified file path.

    Args:
        data (np.array): The NumPy array to be saved.
        path (Path): The file path where the array should be saved.

    Returns:
        None: The function does not return anything; it writes the array to the file.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    # Save NumPy array to file
    with open(path, 'wb') as file:
        np.save(path, data)


def load_numpy_array(path: Path):
    """
    load a NumPy array to a specified file path.

    Args:
        path (Path): The file path.

    Returns:
        None: The function does not return anything.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    # Save NumPy array to file
    with open(path, 'rb') as file:
        return np.load(file)
    
    
def save_object(model, path: Path):
    """
    Saves a Python object (e.g., a machine learning model) to a specified file path using Pickle.

    Args:
        model (object): The object to be saved.
        path (Path): The file path where the object should be saved.

    Returns:
        None: The function does not return anything; it writes the object to the file.
    """
    # Ensure the directory exists
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    # Save the object to file
    with open(path, 'wb') as file:
        pickle.dump(model, file)



