from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.evaluate import evaluate_model

def test_models(features_train, target_train, features_test, target_test, run=True):
    """
    Trains different models on the training set and evaluates them on the test set.
    :param features_train: The features used for training.
    :param target_train: The target data used for training.
    :param features_test: The features used for testing.
    :param target_test: The target data used for testing.
    """

    if not run: return

    models = {
        'lr': LogisticRegression(random_state=42),
        'nb': GaussianNB(),
        'knn': KNeighborsClassifier(),
        'svm': SVC(random_state=42),
        'rfc': RandomForestClassifier(random_state=42),
        'gbc': GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        print(f"\n--- Testing {name} ---")
        model.fit(features_train, target_train)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_test)[:,1]
        else:
            probs = model.decision_function(features_test)
            probs = (probs - probs.min()) / (probs.max() - probs.min())

        evaluate_model(model.predict(features_test), target_test, probs)
