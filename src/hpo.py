from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def hyperparameter_optimization(features, target, output=True):
    """
    Hyperparameter optimization.
    :param features: The features used for optimization.
    :param target: The target data used for optimization.
    :param output: Print the optimization results or not.
    :return: The optimized model.
    """

    if output: verbose = 2
    else: verbose = 0

    param_grid = {
        'n_estimators': [250, 300, 350],
        'learning_rate': [0.02, 0.04, 0.05, 0.06],
        'subsample': [0.75, 0.8, 0.85],
        # set depth and sample_split set after first run through
        'max_depth': [4],
        'min_samples_split': [2],
    }

    gbc = GradientBoostingClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=gbc,
        param_grid=param_grid,
        scoring='average_precision',
        cv=3,
        n_jobs=-1,
        verbose=verbose
    )

    grid_search.fit(features, target)

    print("Best params: ", grid_search.best_params_)
    print("Best PR AUC: ", grid_search.best_score_)

    return grid_search.best_estimator_
