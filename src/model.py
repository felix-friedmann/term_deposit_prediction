from sklearn.linear_model import LogisticRegression

def train_model(features, target):
    """
    Trains a logistic regression on the train split and returns the model.
    :param features: Dataframe containing the features (train split).
    :param target: Dataframe of the target data (train split).
    :return: The trained model.
    """

    # TODO: Hyperparameter-Tuning
    model = LogisticRegression(max_iter=500)
    model.fit(features, target)

    return model
