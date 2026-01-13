## Vorhersage von Festgeldakquisitionen

`main.py` - Datenimport und Aufruf der benötigten Funktionen für Training und Bewertung des Modells.  
`src/eda.py` - Durchführung der explorativen Datenanalyse und Datenbereinigung.  
`src/model.py` - Training verschiedener Modelle.  
`src/evaluate.py` - Bewertung der verschiedenen Modelle und Thresholds.  
`src/hpo.py` - Hyperparametertuning des besten Modells.

---

### Erkenntnisse der explorativen Datenanalyse

Die verwendeten Daten stammen aus einem Marketing-Datensatz einer portugiesischen Bank. [^1]

#### Numerische Features

Das Feature `balance` hat eine sehr breite Streuung und ist stark rechtsschief, hat jedoch vermutlich eine eher geringe Trennkraft. Das Feature
`duration` dagegen weist eine bessere Trennkraft auf, allerdings ist die Dauer eines Anrufs erst nach dem Anruf bekannt, wenn auch schon das Ergebnis bekannt ist.
Um das Modell realistisch zu halten wird das Feature vor dem Modelltraining entfernt. Dem bereinigten Feature `pdays` nach nehmen Kunden, deren letzter Marketingkontakt
kürzer her sind, das Angebot öfter an. Allerdings bietet sich für das Feature (meines Wissenstands nach) keine sinnvolle Bereinigung durch die es möglich wäre das Feature mit in das
Training aufzunehmen, weshalb auch dieses Feature entfernt wird.

<div style="display: flex; justify-content: center; align-items: center; gap: 20px">
    <img src="graphs/duration_feature.png" alt="Diagramm" style="width: 275px">
    <img src="graphs/pdays_feature.png" alt="Diagramm" style="width: 275px">
</div>

Die Features `age`, `day_of_week`, `campaign` und `previous` weisen grundsätzlich keine große Besonderheit auf.

#### Kategorische Features
Durch die Daten wird sichtbar, dass die Akquisitionsrate bei Kunden die Studenten oder im Ruhestand sind höher ist als bei anderen Kategorien des Features `job`. Außerdem 
zeigt auch das Feature `month` deutliche Diskrepanzen in der Akquisitionsrate, wobei in den Monaten März, Oktober, September und Dezember der Anteil der Zusagen
deutlich über dem der restlichen Monate liegt. Das Feature `poutcome` zeigt eine deutliche retention rate, bei der über 60% der Kunden, bei denen frühere Marketingkampagnen
erfolgreich waren, erneut ein Festgeldangebot annehmen. 

<div style="display: flex; justify-content: center; align-items: center; gap: 20px">
    <img src="graphs/month_feature.png" alt="Diagramm" style="width: 275px">
    <img src="graphs/poutcome_feature.png" alt="Diagramm" style="width: 275px">
</div>

Bei den übrigen kategorischen Features `marital`, `education`, `default`, `housing`, `loan` und `contact` zeigen sich keine so deutlichen Diskrepanzen zwischen den einzelnen Kategorien.

Die Zielvariable ist sehr unbalanciert, da nur 11,69% der Fälle zu `yes` gehören, was zur Folge hat, dass die Accuracy verzerrt ist und PR-AUC aussagekräftiger ist, da explizit die Performance auf der
positiven Klasse gemessen wird.

---

### Zielmetrik

Als Zielmetrik wird hauptsächlich die Precision verwendet, also $\frac{TP}{TP+FP}$. Grund dafür ist, dass in dem Fall der Vorhersage von Festgeldakquisitionen
der Anteil von False Positives möglichst gering gehalten werden sollte, während der Anteil von False Negatives nicht ein so hohes Gewicht hat. Ist der Anteil
von False Positives hoch, steigt der Anteil der Kosten die schlussendlich keinen Ertrag bringen. False Negative dagegen sind ausschließlich Opportunitätskosten.
Außerdem werden ROC-AUC Score und PR-AUC Score bei der Modellauswahl verwendet, um die Modelle unabhängig vom Threshold zu vergleichen.

---

### Modellauswahl

Da es sich um ein Klassifikationsproblem handelt, werden als mögliche Modelle die logistische Regression, der Naive Bayes, k nearest neighbors, SVM, Random Forest und Gradient Boosting Classifier in die Auswahl mit aufgenommen.
Unter den genannten Modellen erzielt GBC in der Baseline die besten Ergebnisse mit einem ROC-AUC Score von 0,8 und einem Precision Score von 0,66, weshalb das Modell für weitere Optimierung ausgewählt wurde.

Die Hyperparameter wurden mithilfe eines GridSearchCV auf dem Training-Set optimiert, wobei strukturelle Hyperparameter auf einem groben Grid untersucht wurden, während die restlichen Parameter in einem weiteren Durchlauf weiter
optimiert wurden:  

`learning rate`: 0,04  
`max depth`: 4  
`min sample split`: 2  
`n estimators`: 350  
`subsample`: 0,8

---

### Performance

Die Performance des Modells ist von dem jeweils gewählten Threshold abhängig. Aufgrund der zuvor genannten Priorität der Zielmetrik ist der Optimalpunkt um einen Threshold von 0,6 herum.

| Threshold |  0,5  |  0,6  |  0,7  |
|-----------|:-----:|:-----:|:-----:|
| Precision | 0,740 | 0,825 | 0,890 |
| Recall    | 0,288 | 0,206 | 0,130 | 

Je nachdem welche Priorität gewählt wird, und wie hoch die Kosten pro Anruf beziehungsweise der Ertrag pro erfolgreichen Anruf ist, sollte ein Threshold gewählt werden der die Opportunitätskosten von nicht getätigten Anrufen, und die Kosten von Anrufen ohne Erfolg minimiert. 
Grundsätzlich kann man sagen, dass das Modell in der derzeitigen Form zur Unterstützung von Entscheidungen verwendet werden kann, aber ein geeigneter Threshold anhand realer Daten gewählt werden sollte.

---

### Weiterführend

Weitere Features die zur Prognose sinnvoll sein könnten wäre etwa die Häufigkeit von Ein- und Auszahlungen. Ein Kunde dessen Kontostand volatil ist, wird vermutlich eher seltener ein Festgeldangebot annehmen.
Für einen Kunden, der das Konto de facto bereits als Festgeldkonto nutzt, ist das Angebot dagegen attraktiver. Zusätzliche Variablen, die eine Rolle spielen, könnten wären etwa die aktuelle Wirtschaftslage oder auch
der Konkurrenzmarkt. Ist die aktuelle Kombination aus angebotenem Zins und Inflation attraktiv, werden vermutlich mehr Kunden ein Festgeldangebot annehmen, bietet eine andere Bank einen attraktiveren Zinssatz, sinkt die Attraktivität
von Festgeldkonten der eigenen Bank. Außerdem könnte wie im vorigen Abschnitt erwähnt die Kosten pro Anruf und der Gewinn pro Kunde mit einfließen, um den Threshold zu optimieren.

Unter Umständen kann das Modell auch für andere Bankprodukte verwendet werden. Dafür müsste die Zielvariable angepasst werden, und das Modell erneut trainiert werden, da Features in anderen Bereichen nicht unbedingt die gleiche Relevanz haben
wie bei Festgeldakquisitionen. Voraussetzung dafür ist natürlich, dass die neue Zielvariable entweder für die gleichen Kunden erhoben wird, oder alle Features für neue Kunden erhoben werden. Die jetzige Zielvariable könnte dabei zu einem neuen Feature werden.

[^1]: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.