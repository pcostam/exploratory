# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:37:08 2020

@author: anama
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from preprocessing.series import create_dataset
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

#ver https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/ch04.html
#https://towardsdatascience.com/anomaly-detection-with-isolation-forest-visualization-23cd75c281e2
#https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770
#https://medium.com/@petehouston/set-index-for-dataframe-in-pandas-55400e306e42
def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss


def scatterPlot(xDF, yDF, algoName):
    tempDF = pd.DataFrame(data=xDF.loc[:,0:1], index=xDF.index)
    tempDF = pd.concat((tempDF,yDF), axis=1, join="inner")
    tempDF.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label", data=tempDF, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Separation of Observations using "+algoName)
    
def do_PCA(X_train):
    n_components = 3
    whiten = False
    random_state = 2018

    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    X_train_PCA = pca.fit_transform(X_train)
    X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)
    
    X_train_PCA_inverse = pca.inverse_transform(X_train_PCA)
    X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, \
                                   index=X_train.index)

    anomalyScoresPCA = anomalyScores(X_train, X_train_PCA_inverse)
    #preds = plotResults(y_train, anomalyScoresPCA, True)
    print("anomalyScoresPCA", anomalyScoresPCA)
    

    
def test():
    val1 = create_dataset("sensortgmeasure", str(4), limit=True)["value"]
    val1 = val1.rename(columns={"value": "value1"})
    val2 = create_dataset("sensortgmeasure", str(5), limit=True)["value"]
    val2 = val1.rename(columns={"value": "value2"})
    val3 = create_dataset("sensortgmeasure", str(6), limit=True)["value"]
    val3 = val1.rename(columns={"value": "value3"})
    val4 = create_dataset("sensortgmeasure", str(7), limit=True)["value"]
    val4 = val1.rename(columns={"value": "value4"})
    
    frames = [val1, val2, val3, val4]
    result = pd.concat(frames, axis=1, sort=False)
    result.index = [x for x in range(1, len(result.values)+1)]
    
   
    
    msk = np.random.rand(len(result)) < 0.8
    dataset_train = result[msk]
    dataset_test = result[~msk]
    
   
    scaler = StandardScaler()
    
    #normalize
    X_train = pd.DataFrame(data = scaler.fit_transform(dataset_train),
                           columns = dataset_train.columns,
                           index = dataset_train.index)
    
    X_test = pd.DataFrame(data = scaler.transform(dataset_test),
                          columns = dataset_test.columns,
                          index = dataset_test.index)
    
   
    
    do_PCA(X_train)
    return True