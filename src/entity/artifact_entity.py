from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_path: str
    test_path: str
    raw_path: str
