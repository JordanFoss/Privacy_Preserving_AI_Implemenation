import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def plotFeatureImportancesDiabetes(model, diabetes_features):
    """
    This function plots the importance of each of the features in the diabetes
    data set. This is used for the DT and RF models as they offer the 
    functionality.

    Parameters
    ----------
    model : sklearn.ensemble.[model type]
        The model that is being used to plot.
    diabetes_features : list(str)
        The features that are present in the diabetes data set.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

def loadDiabetesData():
    """
    This function loads the diabetes data set and normalises it.

    Returns
    -------
    diabetes : Pandas.Dataframe
        Normalised diabetes data set as a Pandas dataframe.

    """
    diabetes = pd.read_csv('/home/jordan/Documents/UniversityStuff/2022/Semester2/ENGG4812/Privacy_Preserving_AI_Implemenation/Data_Sets/diabetes.csv')
    pd.set_option('mode.chained_assignment', None)
    
    #Min-Max Normalise the data
    scaler = MinMaxScaler()
    scaler.fit(diabetes)
    scaled = scaler.fit_transform(diabetes)
    diabetes = pd.DataFrame(scaled, columns=diabetes.columns)
    return diabetes

def addLaplaceNoise(dataSet, mu, scale):
    """
    This function takes a pandas dataframe and added random noise to it from
    the Laplace distribution. The parameters os said distribution are defined
    in the functin call.
    
    Parameters
    ----------
    dataSet : Pandas.Dataframe
        The dataset that you wish to add the noise to.
    mu : float
        The mean for the laplace distribution.
    scale : float
        The expoential decay, which is effective the standard deviation.

    Returns
    -------
    privateDataSet : Pandas.Dataframe
        A copy of the dataSet with laplace random noise added.

    """
    privateDataSet = dataSet.copy()
    for feature in dataSet.columns:
        for index in range(dataSet[feature].size):
            privateDataSet[feature][index] += round(np.random.laplace(mu, scale))
    return privateDataSet
    
def addGaussianNoise(dataSet, mu, scale):
    """
    This function takes a pandas dataframe and added random noise to it from
    the Gaussian distribution. The parameters os said distribution are defined
    in the functin call.
    
    Parameters
    ----------
    dataSet : Pandas.Dataframe
        The dataset that you wish to add the noise to.
    mu : float
        The mean for the laplace distribution.
    scale : float
        The standard deviation.

    Returns
    -------
    privateDataSet : Pandas.Dataframe
        A copy of the dataSet with gaussian random noise added.

    """
    privateDataSet = dataSet.copy()
    for feature in dataSet.columns:
        for index in range(dataSet[feature].size):
            privateDataSet[feature][index] += round(np.random.normal(mu, scale))
    return privateDataSet


