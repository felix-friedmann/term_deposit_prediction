from ucimlrepo import fetch_ucirepo
from src.eda import run_eda
from src.model import train_model
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

"""
Vorhersage von Festgeldakquisitionen

`main.py` - Datenimport und Aufruf der benötigten Funktionen für Training und Bewertung des Modells.  
`src/eda.py` - Durchführung der explorativen Datenanalyse und Datenbereinigung.  
`src/model.py` - Training einer Logistic Regression auf dem Test-Split.  
`src/evaluate.py` - Bewertung des trainierten Modells anhand von Precision und Accuracy. 
"""
def main():
    # fetch dataset from UCI repository
    bank_marketing = fetch_ucirepo(id=222)

    # data (as pandas dataframes)
    features = bank_marketing.data.features
    target = bank_marketing.data.targets

    # running exploratory data analysis (check distributions, missing values, etc.)
    cleaned_features, cleaned_target = run_eda(features, target, False)

    # splitting the dataset in train and test, keeping the proportions of yes/no in target
    features_train, features_test, target_train, target_test = train_test_split(
        cleaned_features, cleaned_target, test_size=0.2, random_state=42, stratify=cleaned_target
    )

    # training the chosen model (logistic regression) on the training data
    model = train_model(features_train, target_train)

    # evaluating the trained model on the test data with ...
    evaluate_model(target_test, model.predict(features_test))

if __name__ == '__main__':
    main()
