from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements]  # Use strip() to remove newlines and spaces

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. No requirements will be installed.")
    
    return requirements

setup(
    name="credit_card_fraud_detection",  # Changed to a more standard format
    version="0.0.1",
    description="Predicting whether a credit card transaction is fraudulent or not",
    author="Pavan Bairu",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    long_description="This package provides tools for detecting fraudulent credit card transactions using machine learning.",
    license="MIT",  # Specify your license
)
