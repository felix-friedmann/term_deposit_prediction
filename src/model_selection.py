from sklearn.model_selection import cross_validate
from src.config import MODELS
import logging

def test_models(features_train, target_train):
    """
    Trains different models on the training set and evaluates them with a certain cross validation.
    :param features_train: The features used for training.
    :param target_train: The target data used for training.
    :return: List of testing results.
    """

    logger = logging.getLogger(__name__)

    scoring = {
        'pr_auc': 'average_precision',
        'auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
    }

    results = {}

    for name, model in MODELS.items():

        logger.info(f'Testing model {name}...')

        cv = cross_validate(
            model,
            features_train,
            target_train,
            scoring=scoring,
            cv=3
        )

        results[name] = {
            metric: round(float(cv[f"test_{metric}"].mean()), 4)
            for metric in scoring.keys()
        }

    return results
