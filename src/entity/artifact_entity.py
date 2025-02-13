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
