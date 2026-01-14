from ucimlrepo import fetch_ucirepo
from src.eda import run_eda
from src.eda import preprocessing
from src.model_selection import test_models
from src.evaluate import evaluate_thresholds
from src.hpo import hyperparameter_optimization
from sklearn.model_selection import train_test_split
import argparse

"""
Term deposit prediction

`main.py` - Import of data and calling of needed functions for EDA, model selection and HPO of the best model.  
`src/config.py` - Initializing the model dictionaries.
`src/eda.py` - Executing exploratory data analysis and data cleanup.  
`src/evaluate.py` - Evaluation of a given model and of different thresholds.  
`src/hpo.py` - Hyperparametertuning of the best model.  
`src/model.py` - Training of different models.
"""
def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--eda', action='store_true', help="Runs exploratory data analysis.")
    parser.add_argument('--evaluate_models', action='store_true', help="Tests different models and prints results.")
    parser.add_argument('--model', type=str, default='gbc', choices=['lr', 'nb', 'knn', 'svm', 'rfc', 'gbc'], help="Specify the model to run HPO and threshold evaluation on. Default: gbc.")
    parser.add_argument('--train', action='store_true', help="Runs HPO and threshold evaluation.")
    args = parser.parse_args()

    # fetch dataset from UCI repository
    bank_marketing = fetch_ucirepo(id=222)

    # data (as pandas dataframes)
    features = bank_marketing.data.features
    target = bank_marketing.data.targets

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

    # running exploratory data analysis (check distributions, missing values, etc.)
    if args.eda:
        run_eda(features, target)

    # evaluate different models
    if args.evaluate_models:
        results = test_models(features_train_clean, target_train_clean)
        for model, metric in results.items():
            print(model, metric)

    # HPO for chosen model
    if args.train:
        selected_model = args.model
        print(f"\nRunning HPO and threshold evaluation on {selected_model.upper()}")
        best_model = hyperparameter_optimization(features_train_clean, target_train_clean, model_name=selected_model)

        # evaluate best thresholds
        probs = best_model.predict_proba(features_test_clean)[:, 1]
        evaluate_thresholds(target_test_clean, probs)

if __name__ == '__main__':
    main()
