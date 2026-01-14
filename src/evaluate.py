from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

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

    print(f"\n--- Independent of thresholds ---")
    print(f"ROC AUC: {roc_auc_score(true_values, probs)}")
    print(f"PR AUC: {average_precision_score(true_values, probs)}")
