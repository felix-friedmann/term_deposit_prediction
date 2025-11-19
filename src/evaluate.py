from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

def evaluate_model(true_values, predicted_values):
    """
    Evaluates the model at hand with help of precision and recall as well as F1 score by printing them.
    :param true_values: The target data which is part of the test split.
    :param predicted_values: The predicted values, resulting from running the model on the feature data of the test split.
    """

    print(f"\nPrecision score: {precision_score(true_values, predicted_values)}")
    print(f"Recall score: {recall_score(true_values, predicted_values)}")
    print(f"Accuracy: {accuracy_score(true_values, predicted_values)}")
