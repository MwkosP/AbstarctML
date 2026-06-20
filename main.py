
#downloadDataset(name="Hello-SimpleAI/HC3",source="huggingface",outputPath="Datasets",saveAs="hc3_raw",revision="refs/convert/parquet",)

# RAID is large; uncomment when you are ready to download the full raw dataset.
# downloadDataset(name="liamdugan/raid",source="huggingface",outputPath="Datasets",saveAs="raid_raw")

'''

from Data import describeData, downloadDataset, getCorrelations, getDatasetInfo, getMissingness, getOutliers, listAvailableDatasets, plotCorrelations, plotDataset, plotDistributions, plotMissingness, plotOutliers, setActiveDataset, viewHead, viewSample, viewSchema
#downloadDataset(name="iris",source="sklearn",saveAs="iris_raw")

setActiveDataset(name="iris_raw")
print(describeData())



'''


from Data import describeData, downloadDataset, getCorrelations, getDatasetInfo, getMissingness, getOutliers, listAvailableDatasets, plotCorrelations, plotDataset, plotDistributions, plotMissingness, plotOutliers, setActiveDataset, viewHead, viewSample, viewSchema
#downloadDataset(name="digits",source="sklearn",split="train[:100]",outputPath="Datasets",saveAs="digits_raw",)
Dataset = "digits_raw"
setActiveDataset(Dataset)



print(describeData())
#plotDataset(dimensions=2,method="pca",colorBy="target",columns=None,sample=5000,outputPath="Cache/EDA/dataset_2d.png",show=True)
plotDataset(dimensions=3,method="tsne",colorBy="target",columns=None,sample=1000,outputPath="Cache/EDA/dataset_3d.png",show=True)
#plotDistributions(columns=None,outputDir="Cache/EDA",show=True)
#plotCorrelations(outputPath="Cache/EDA/correlations.png",show=True)
#plotMissingness(outputPath="Cache/EDA/missingness.png",show=True)
#plotOutliers(columns=None,outputDir="Cache/EDA",show=True)


