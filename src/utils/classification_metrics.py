import os
import sys
from src.exception.exception import CreditFraudException
from src.logger.logging import logging
from src.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


def get_classification_scores(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Compute classification metrics (accuracy, F1 score, precision, and recall) 
    and return them as a ClassificationMetricArtifact object.

    Args:
        y_true (array-like): Ground truth (actual) target values.
        y_pred (array-like): Predicted target values.

    Returns:
        ClassificationMetricArtifact: An object containing accuracy, F1 score, precision, and recall metrics.
    """
    try:

        # Compute individual classification metrics
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        # Create and return the classification metrics artifact
        classification_metrics_artifact = ClassificationMetricArtifact(
            accuracy=accuracy,
            f1_score=f1,
            precision_score=precision,
            recall_score=recall
        )

        return classification_metrics_artifact

    except Exception as e:
        logging.error(f"Error while calculating classification metrics. Error: {e}")
        raise CreditFraudException(e, sys)
