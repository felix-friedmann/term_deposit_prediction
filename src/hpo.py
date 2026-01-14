from sklearn.model_selection import GridSearchCV
from src.config import MODELS, HPO_PARAM_GRID
import pandas as pd
import logging

def hyperparameter_optimization(features, target, model_name):
    """
    Hyperparameter optimization.
    :param features: The features used for optimization.
    :param target: The target data used for optimization.
    :param model_name: The model to optimize.
    :return: The optimized model.
    """

    logger = logging.getLogger(__name__)

    grid_search = GridSearchCV(
        estimator=MODELS[model_name],
        param_grid=HPO_PARAM_GRID[model_name],
        scoring='average_precision',
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(features, target)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best PR AUC: {grid_search.best_score_}")

    return grid_search.best_estimator_
