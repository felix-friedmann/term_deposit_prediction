from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score


def evaluate_model(features_test, true_values, model):
    """
    Evaluates the model at hand with help of precision and recall as well as F1 score by printing them.
    :param features_test: The features used for prediction.
    :param true_values: The target data which is part of the test split.
    :param model: The model to be evaluated.
    """

    prob = model.predict_proba(features_test)[:, 1]

    # test different thresholds because dataset is unbalanced
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        predicted = (prob >= threshold).astype(int)
        print(f"\n--- Threshold: {threshold} ---")
        print(f"Precision: {precision_score(true_values, predicted)}")
        print(f"Recall: {recall_score(true_values, predicted)}")
        print(f"Conf Matrix: {confusion_matrix(true_values, predicted)}")
        print(f"Accuracy: {accuracy_score(true_values, predicted)}")
