## Vorhersage von Festgeldakquisitionen

`main.py` - Datenimport und Aufruf der benötigten Funktionen für Training und Bewertung des Modells.  
`src/eda.py` - Durchführung der explorativen Datenanalyse und Datenbereinigung.  
`src/model.py` - Training einer Logistic Regression auf dem Test-Split.  
`src/evaluate.py` - Bewertung des trainierten Modells anhand von Precision und Accuracy.  

---

### Erkenntnisse der explorativen Datenanalyse

Die verwendeten Daten stammen aus einem Marketing-Datensatz einer portugiesischen Bank. [^1]

#### Numerische Features

Das Feature `balance` hat eine sehr breite Streuung und ist stark rechtsschief, hat jedoch vermutlich eine eher geringe Trennkraft. Das Feature
`duration` dagegen weist eine bessere Trennkraft auf, allerdings ist die Dauer eines Anrufs erst nach dem Anruf bekannt, wenn auch schon das Ergebnis bekannt ist.
Um das Modell realistisch zu halten wird das Feature vor dem Modelltraining entfernt. Dem bereinigten Feature `pdays` nach nehmen Kunden, deren letzter Marketingkontakt
kürzer her sind, das Angebot öfter an. Allerdings bietet sich für das Feature (meines Wissenstands nach) keine sinnvolle Bereinigung wodurch es möglich wäre das Feature mit in das
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

---

### Zielmetrik

Als Zielmetrik wird hauptsächlich die Precision verwendet, also $\frac{TP}{TP+FP}$. Grund dafür ist, dass in dem Fall der Vorhersage von Festgeldakquisitionen
der Anteil von False Positives möglichst gering gehalten werden sollte, während der Anteil von False Negatives nicht ein so hohes Gewicht hat. Ist der Anteil
von False Positives hoch, steigt der Anteil der Kosten die schlussendlich keinen Ertrag bringen. False Negative dagegen sind ausschließlich Opportunitätskosten.

Zusätzlich wird zur allgemeinen Bewertung des Modells die Accuracy herangezogen, um generell darzustellen, welchen Anteil an Fällen das Modell korrekt erkennt.

---

### Performance

Die Performance des Modells kann eher als mittelmäßig beschrieben werden. Der Precision Score liegt bei 66,31% was bedeutet, dass der Anteil der Kunden für den vorhergesagt wird, dass er das Angebot annimmt,
in nur knapp 66% der Fälle auch wirklich das Angebot annimmt. Die Accuracy dagegen liegt bei etwa 89%. Grund dafür ist die unausgeglichene Verteilung der Antworten, die `yes` Kategorie, bildet nur 11,69% der Fälle.
Eine Möglichkeit das Modell zu verbessern wäre Hyperparameter-Tuning, und im Idealfall ein höherer Anteil an erfolgreichen Akquisitionen im Datensatz.

Zusammenfassend kann man sagen, dass das Modell in der derzeitigen Form zur Unterstützung von Entscheidungen verwendet werden kann, aber nicht immer optimalen Ergebnisse liefert. Zuvor wäre noch Hyperparameter-Tuning sinnvoll, 
und auch eine laufende Erweiterung des Datensatzes ist empfehlenswert.

---

### Weiterführend

_TODO_

[^1]: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.