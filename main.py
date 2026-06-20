from Data import createDatasetSubset, describeData, downloadDataset, getCorrelations, getDatasetInfo, getMissingness, getOutliers, listAvailableDatasets, plotCorrelations, plotDataset, plotDistributions, plotMissingness, plotOutliers, setActiveDataset, viewHead, viewSample, viewSchema,dropMissing
#listAvailableDatasets()
#downloadDataset(name="liamdugan/raid",source="huggingface",outputPath="Datasets",saveAs="raid_raw")
#createDatasetSubset(name="raid_raw",saveAs="raid_train_100k_balanced",sample=100000,split="train",stratifyBy=["label", "domain"],balance=True,randomState=42)

Dataset = "raid_train_100k_balanced"
setActiveDataset(Dataset)





#------------EDA----------------
#describeData()  # or sample=10000
#viewHead(n=10)
#viewSample(n=20)
#viewSchema()
#plotDataset(dimensions=2,method="pca",colorBy="target",columns=None,sample=5000,outputPath="Cache/EDA/dataset_2d.png",show=True)
#plotDataset(dimensions=3,method="tsne",colorBy="source",columns=None,sample=1000,outputPath="Cache/EDA/dataset_3d.png",show=True)
#plotDistributions(columns=None,outputDir="Cache",show=True)
#plotCorrelations(outputPath="Cache/correlations.png",show=True)
#plotMissingness(outputPath="Cache/Missingness.png",show=True)
#plotOutliers(columns=None,outputDir="Cache/Outliers.png",show=True)



#------------Preprocessing----------------
#preprocessing.py
#scaleData(method="standard",columns=None)
#encodeData(method="onehot",columns=None)
#imputeMissing(method="mean",columns=None,value=None)
#dropMissing(columns=None,threshold=None) 
#encodeTarget(y,method="label")



#------------Feature Engineering----------------
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