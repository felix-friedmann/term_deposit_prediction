from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
import pandas as pd
import logging

def evaluate_thresholds(true_values, probs):
    """
    Evaluates the different thresholds.
    :param true_values: The target data which is part of the test split.
    :param probs: The probabilities of the predicted values of the test split.
    """

    logger = logging.getLogger(__name__)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    logger.info("Returning threshold evaluation")
    for threshold in thresholds:
        predicted = (probs >= threshold).astype(int)
        results.append({
            "Threshold": threshold,
            "Precision": precision_score(true_values, predicted),
            "Recall": recall_score(true_values, predicted),
        })

    print(results)

    logger.info("Evaluation independent of thresholds")
    print(
        pd.DataFrame({
            "ROC AUC": roc_auc_score(true_values, probs),
            "PR AUC": average_precision_score(true_values, probs),
        })
    )
