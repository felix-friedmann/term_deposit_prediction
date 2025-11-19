import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, RobustScaler

def run_eda(features, target, output):
    """
    Performing descriptive statistics to understand the underlying data, cleanup of non-used variables, converting of categorical features and scaling of numeric ones.
    :param features: Dataframe containing the features.
    :param target: Dataframe of the target data.
    :param output: Controls if descriptive statistics should be printed or not.
    :return: The cleaned feature and target data.
    """

    df_num = features.select_dtypes(include='int64')
    df_cat = features.select_dtypes(include='object')

    if output:
        # general distribution of the target variable
        print(f"Distribution of target variables: {target.value_counts(normalize=True)}")

        # descriptive statistics for the numeric features
        for col in df_num:
            yes_values = df_num[target['y'] == 'yes'][col]
            no_values = df_num[target['y'] == 'no'][col]

            # cleaning up 'pdays' because the categorical value -1 disturbs the descriptive statistics
            if col == 'pdays':
                yes_values = yes_values.replace(-1, np.nan).dropna()
                no_values = no_values.replace(-1, np.nan).dropna()

            plt.boxplot([yes_values, no_values], labels=['yes', 'no'])
            plt.title(col)
            plt.show()

            print(f"\nYes stats '{col}':")
            print(yes_values.describe())

            print(f"\nNo stats '{col}':")
            print(no_values.describe())

        # descriptive statistics for the categorical features
        for col in df_cat:
            table = pd.crosstab(index=df_cat[col], columns=target['y'], margins=True, normalize='index')
            table = table.drop("All", axis='index')

            print(f"\nColumn: {col}")
            print(table)

            table.plot(kind='bar')
            plt.title(col)
            plt.ylabel("Proportion")

            plt.tight_layout()
            plt.show()

    # Dropping the variable 'duration' because it is not known before the call and 'pdays' because there is no clean interpretation and bias through the value -1.
    # It is a possibility to add an extra binary feature and drop -1, but logistic regression can't handle nans. The feature 'previous' already has the implicit
    # binary value, so there shouldn't be a big information loss.
    df_num = df_num.drop(columns=['duration', 'pdays'])

    # scaling numerical features, robust to outliers (needed for 'balance' and 'previous')
    scaler = RobustScaler()
    df_num_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns, index=df_num.index)

    # converting categorical features
    encoder = OneHotEncoder(sparse_output=False)
    arr_enc = encoder.fit_transform(df_cat)
    df_enc = pd.DataFrame(arr_enc, columns=encoder.get_feature_names_out(df_cat.columns), index=df_cat.index)
    clean_features = pd.concat([df_num_scaled, df_enc], axis='columns')

    # converting target to binary
    clean_target = target['y'].map({'yes': 1, 'no': 0})

    return clean_features, clean_target
