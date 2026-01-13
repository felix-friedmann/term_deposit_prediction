## Term deposit prediction

`main.py` - Import of data and calling of needed functions for EDA, model selection and HPO of the best model.  
`src/eda.py` - Executing exploratory data analysis and data cleanup.  
`src/model.py` - Training of different models.  
`src/evaluate.py` - Evaluation of a given model and of different thresholds.  
`src/hpo.py` - Hyperparametertuning of the best model.

---

### EDA insights

The used data is part of a marketing dataset of a portuguese bank. [^1]

#### Numerical Features

The feature `balance` shows a very wide distribution and is strongly right-skewed but likely has low predictive power. The feature `duration` exhibits better discrimination, 
however, call duration is only known after the call, which would make its inclusion unrealistic. Therefore, duration is removed before training.
The cleaned feature `pdays` indicates that customers contacted more recently tend to accept the offer more often. However, no practical preprocessing allows its meaningful 
inclusion, so it is removed as well.

<div style="display: flex; justify-content: center; align-items: center; gap: 20px">
    <img src="graphs/duration_feature.png" alt="Diagramm" style="width: 275px">
    <img src="graphs/pdays_feature.png" alt="Diagramm" style="width: 275px">
</div>

The features `age`, `day_of_week`, `campaign` and `previous` show no remarkable patterns.

#### Categorical Features
The dataset reveals that subscription rates are higher for customers who are students or retired than other categories of the `job` feature. The feature `month` also shows notable 
discrepancies: in March, October, September, and December, the proportion of positive responses is significantly higher than in other months. The feature `poutcome` highlights retention, 
as more than 60% of customers with previous successful campaigns accept the offer again.

<div style="display: flex; justify-content: center; align-items: center; gap: 20px">
    <img src="graphs/month_feature.png" alt="Diagramm" style="width: 275px">
    <img src="graphs/poutcome_feature.png" alt="Diagramm" style="width: 275px">
</div>

The other categorical features `marital`, `education`, `default`, `housing`, `loan` and `contact` show no substantial differences between categories.

The target variable is unbalanced as only 11.69% of cases have a `yes` result. The consequence is that the Accuracy is distorted and PR-AUC is more meaningful because the performance of the
positive class is explicitly measured.

---

### Target Metric

The primary target metric is Precision because false positives are costly in term deposit prediction, whereas false negatives represent only opportunity costs. High false positives 
increase costs without generating revenue, while false negatives are less critical. Additionally, ROC-AUC and PR-AUC are used during model selection because they allow comparison of 
models independent of any threshold.

---

### Model Selection

Since this is a classification problem, the candidate models include Logistic Regression, Naive Bayes, k-Nearest Neighbors, SVM, Random Forest, and Gradient Boosting Classifier.
Among these, GBC achieves the best baseline performance with a ROC-AUC of 0.80 and Precision of 0.66, making it the model selected for further optimization.
Hyperparameters were tuned using GridSearchCV on the training set. Structural parameters (`max_depth`, `min_samples_split`) were explored on a coarse grid, while optimization-related 
parameters (`learning_rate`, `n_estimators`, `subsample`) were fine-tuned in a second pass:

`max_depth`: 4  
`min_sample_split`: 2  
`learning_rate`: 0.04  
`n_estimators`: 350  
`subsample`: 0.8

---

### Performance

Model performance depends on the chosen decision threshold. Given the priority on Precision, the optimal threshold lies around 0.6.

| Threshold |  0.5  |  0.6  |  0.7  |
|-----------|:-----:|:-----:|:-----:|
| Precision | 0.740 | 0.825 | 0.890 |
| Recall    | 0.288 | 0.206 | 0.130 | 

The threshold should be chosen considering the cost per call and revenue per successful subscription, balancing opportunity costs and wasted effort. The model in its current form is 
suitable for decision support, but the threshold should ideally be adapted to real-world cost/revenue data.

---

### Further Work

Additional features could improve prediction, such as account deposit/withdrawal frequency. Customers with volatile balances may accept term deposit offers less frequently, while those 
already using the account as a term deposit might be more likely to subscribe. Other potential variables include economic conditions or competitors’ interest rates: if the offered rate 
is attractive relative to inflation or other banks, subscription rates may increase, otherwise, they may decrease. The cost per call and profit per customer could also be incorporated 
into threshold selection.

The model could be adapted for other banking products by adjusting the target variable and retraining. Some features may lose relevance, and a requirement is that the new target variable is
collected from the same customers or all features are selected for new customers. The previous target could even serve as a new feature.

[^1]: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.