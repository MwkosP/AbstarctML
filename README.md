# abstractml

A small, abstracted ML toolkit inspired by sklearn.

The project is organized by purpose: <br/>`Data/` handles datasets, EDA, plots, and preprocessing; <br/>`Models/` handles estimators; <br/>`Evaluation/` handles metrics; <br/>`Core/` holds shared utilities.

## Example

```python
from Data import setActiveDataset, describeData
from Models import selectModel, trainModel
from Evaluation import evaluateModel

setActiveDataset("iris_raw")
print(describeData())

model = selectModel(name="logistic_regression", max_iter=1000)
trained = trainModel(model=model, target="target")
print(evaluateModel(trained=trained))
```

## Status

Early development. See `RULES.md` for structure and conventions.