from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    train_path: Path
    test_path: Path
    raw_path: Path

@dataclass
class DataValidationArtifact:
    valid_train_path: Path
    valid_test_path: Path
    invalid_train_path: Path
    invalid_test_path: Path
    validation_status: Path

@dataclass
class DataTransformationArtifact:
    transformed_train_path: Path
    transformed_test_path: Path
    preprocessor_path: Path

@dataclass
class ClassificationMetricArtifact:
    accuracy: float
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass
class ModelTrainerArtifact:
    model_path: Path
    train_metrics: ClassificationMetricArtifact
    test_metrics: ClassificationMetricArtifact
