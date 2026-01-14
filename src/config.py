from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42

MODELS = {
    'lr': LogisticRegression(random_state=RANDOM_STATE),
    'nb': GaussianNB(),
    'knn': KNeighborsClassifier(),
    'svm': SVC(random_state=RANDOM_STATE, probability=True),
    'rfc': RandomForestClassifier(random_state=RANDOM_STATE),
    'gbc': GradientBoostingClassifier(random_state=RANDOM_STATE)
}

HPO_PARAM_GRID = {
        'lr': {
            'C': [0.01, 0.05, 0.1, 0.5, 1],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        },
        'nb': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'knn': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        },
        'svm': {
            'C': [0.1, 0.5, 1, 5],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'rfc': {
            'n_estimators': [250, 300, 350],
            'max_depth': [None, 5, 7],
            'min_samples_split': [2, 3, 4],
            'max_features': ['sqrt', 'log2']
        },
        'gbc': {
            'n_estimators': [250, 300, 350],
            'learning_rate': [0.02, 0.04, 0.05],
            'subsample': [0.75, 0.8, 0.85],
            'max_depth': [3, 4],
            'min_samples_split': [2, 3],
        }
    }