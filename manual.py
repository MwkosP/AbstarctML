# ____________________ Core ____________________


from Models import listAvailableModels, predictModel, selectModel, trainModel
from Evaluation import evaluateModel, listAvailableMetrics, registerMetric, runMetric


# ____________________ Data ____________________
from Data import addAverageWordLength, addInteraction, addRatio, addRegexCount, addTextLength, addTextStats, addWordCount, binColumn, combineColumns, createDatasetSubset, createFeature, describeData, downloadDataset, dropHighCardinality, dropLowVariance, extractDateParts, getCorrelations, getDatasetInfo, getMissingness, getOutliers, listAvailableDatasets, plotCorrelations, plotDataset, plotDistributions, plotMissingness, plotOutliers, setActiveDataset, viewHead, viewSample, viewSchema
# dataset.py
setActiveDataset("iris_raw")
#downloadDataset(name="nyu-mll/glue",source="huggingface",configName="mnli",split="train[:100]",outputPath="Datasets",saveAs="glue_mnli_sample",)
#createDatasetSubset(name="raid_raw",saveAs="raid_100k_balanced",sample=100000,columns=None,stratifyBy=["label","domain"],balance=True,randomState=42)
#listAvailableDatasets()
#print(getDatasetInfo(name="nyu-mll/glue",configName="mnli"))  #from HF if not laready downloaded.


#eda.py
#describeData(show=True)
#describeData(show=True,sample=10000)
#print(getMissingness())
#print(getCorrelations())
#print(getOutliers())
#print(viewHead(n=5))
#print(viewSample(n=100))
#print(viewSchema())
#plotDistributions(columns=None,outputDir="Cache",show=False)
#plotDataset(dimensions=2,method="pca",colorBy="target",columns=None,sample=5000,outputPath="Cache/dataset_2d.png",show=False)
#plotDataset(dimensions=3,method="pca",colorBy="target",columns=None,sample=5000,outputPath="Cache/dataset_3d.png",show=False)
#plotCorrelations(outputPath="Cache",show=False)
#plotMissingness(outputPath="Cache",show=True)
#plotOutliers(columns=None,outputDir="Cache",show=True)




#preprocessing.py
#scaleData(method="standard",columns=None)  #Scales numeric columns. Useful for models that need normalized numbers.
#encodeData(method="onehot",columns=None)   #Converts categorical/text-like columns into numbers.
#imputeMissing(method="mean",columns=None,value=None) #Fills missing/null values.
#dropMissing(columns=None,threshold=None)    #Removes missing data.
#encodeTarget(y,method="label")   #Encodes the target labels separately.




#featureEngineering.py
#createFeature(name="sepalRatio",expression="sepal_length / sepal_width")
#binColumn(column="sepal_length",bins=3,newColumn="sepalLengthBin")
#combineColumns(columns=["domain","model"],newColumn="domainModel")
#addTextLength(column="generation")
#addWordCount(column="generation")
#addAverageWordLength(column="generation")
#addRegexCount(column="generation",pattern=r"\d",newColumn="generationDigitCount")
#addTextStats(column="generation")
#extractDateParts(column="created_at")
#addInteraction(columns=["sepal_length","sepal_width"],operation="multiply",newColumn="sepalArea")
#addRatio(numerator="sepal_length",denominator="sepal_width",newColumn="sepalRatio")
#dropLowVariance(threshold=0.0)
#dropHighCardinality(columns=None,threshold=1000)






# ____________________ Models ____________________
#listAvailableModels()
#model = selectModel(name="logistic_regression",max_iter=1000)
#trained = trainModel(model=model,target="target",features=None,testSize=0.2,randomState=42)
#print(trained)
#predictions = predictModel(model=trained["model"],features=None)
#print(predictions)

# ____________________ Evaluation ____________________
#listAvailableMetrics(task="classification")
#listAvailableMetrics(task="clustering")
#scores = evaluateModel(trained=trained,task="classification",metrics=None)
#print(scores)
#scores = evaluateModel(yTrue=trained["yTest"],yPred=trained["model"].predict(trained["XTest"]),task="classification",metrics=["accuracy","f1"])
#oneMetric = runMetric(yTrue=trained["yTest"],yPred=trained["model"].predict(trained["XTest"]),metric="accuracy",task="classification")
#clusterScores = evaluateModel(X=trained["XTest"],yPred=trained["model"].predict(trained["XTest"]),task="clustering",metrics=["silhouette","calinski_harabasz","davies_bouldin"])
#registerMetric(name="customAccuracy",metric=lambda yTrue,yPred: (yTrue == yPred).mean(),task="classification")



# ____________________ Tuning ____________________




# ____________________ Pipeline ____________________
