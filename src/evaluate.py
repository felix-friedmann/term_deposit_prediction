from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

def evaluate_model(predicted_values, true_values, probs=None):
    """
    Evaluates the model at hand with help of precision and recall as well as F1 score by printing them.
    :param predicted_values: The predicted values.
    :param true_values: The target data which is part of the test split.
    :param probs: The probabilities of the predicted values of the test split.
    """

    print(f"Precision: {precision_score(true_values, predicted_values)}")
    print(f"Recall: {recall_score(true_values, predicted_values)}")
    print(f"Accuracy: {accuracy_score(true_values, predicted_values)}")

    if probs is not None:
        print(f"ROC AUC: {roc_auc_score(true_values, probs)}")
        print(f"PR AUC: {average_precision_score(true_values, probs)}")

def evaluate_thresholds(true_values, probs):
    """
    Evaluates the different thresholds.
    :param true_values: The target data which is part of the train split.
    :param probs: The probabilities of the predicted values of the train split.
    """

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        predicted = (probs >= threshold).astype(int)
        print(f"\n--- Threshold: {threshold} ---")
        print(f"Precision: {precision_score(true_values, predicted)}")
        print(f"Recall: {recall_score(true_values, predicted)}")
        print(f"Accuracy: {accuracy_score(true_values, predicted)}")