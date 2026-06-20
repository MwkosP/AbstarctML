<div align="center">

# AbstractML
A small, abstracted ML toolkit to make it easy to perform ML Tasks/code.<br/>
<img src="Assets/imgs/AbstractML.png" width="700">
</div>




## Purpose
My entire life i felt like most information on the intertnet and documentations/instructions create more of a mess than to actually explain things. i like to keep things simple. 
<br/>
<br/>
1 simple sentence can beat entire books of information. Same way 1 line functions should be abstracting some ML tasks to hide complecity and Boilerplate code..


The project is organized by purpose: <br/>`Data/` handles datasets, EDA, plots, and preprocessing; <br/>`Models/` handles estimators; <br/>`Evaluation/` handles metrics;<br/>`Tuning/` fine tuning functions/pipelines;  <br/>`Pipeline/` Construct entire Pipeliens of the previosuly listed functions easily with Pipeline/ Module  <br/>`Core/` holds shared utilities.

## Example

```python
from Data import setActiveDataset, describeData
from Models import selectModel, trainModel
from Evaluation import evaluateModel

setActiveDataset("iris_raw") #this says hey we are talking about this Dataset from now on.
print(describeData())

model = selectModel(name="logistic_regression", max_iter=1000)
trained = trainModel(model=model, target="target")
print(evaluateModel(trained=trained))
```

## Status

Early development. See `RULES.md` for structure and conventions.