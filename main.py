from ucimlrepo import fetch_ucirepo
from src.eda import run_eda
from src.eda import preprocessing
from src.model import test_models
from src.evaluate import evaluate_thresholds
from src.hpo import hyperparameter_optimization
from sklearn.model_selection import train_test_split

"""
Vorhersage von Festgeldakquisitionen

`main.py` - Datenimport und Aufruf der benötigten Funktionen für Training und Bewertung des Modells.  
`src/eda.py` - Durchführung der explorativen Datenanalyse und Datenbereinigung.  
`src/model.py` - Training verschiedener Modelle.  
`src/evaluate.py` - Bewertung der verschiedenen Modelle und Thresholds.
`src/hpo.py` - Hyperparametertuning des besten Modells.
"""
def main():
    # fetch dataset from UCI repository
    bank_marketing = fetch_ucirepo(id=222)

    # data (as pandas dataframes)
    features = bank_marketing.data.features
    target = bank_marketing.data.targets

    # running exploratory data analysis (check distributions, missing values, etc.)
    run_eda(features, target, output=False)

    # splitting the dataset in train and test, keeping the proportions of yes/no in target
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    # clean training data
    features_train_clean, target_train_clean, scaler, encoder = preprocessing(
        features_train, target_train
    )

    # clean test data
    features_test_clean, target_test_clean, _, _ = preprocessing(
        features_test, target_test, scaler, encoder, fit=False
    )

    # test models
    test_models(features_train_clean, target_train_clean, features_test_clean, target_test_clean, run=False)

    # HPO for best model (GBC)
    best_model = hyperparameter_optimization(features_train_clean, target_train_clean, output=False)

    # evaluate best thresholds
    probs = best_model.predict_proba(features_train_clean)[:, 1]
    evaluate_thresholds(target_train_clean, probs)

if __name__ == '__main__':
    main()
