import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import random

#Model Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def generateModel(model_type, X_train, y_train):
    """
    This function generates a model and trains it based on the model type.
        
    Parameters:
    ----------
    model_type : str
        Type of model.
    X_train : [float]
        Training data for the model
    y_train : [float]
        Training labels for the model
    Returns
    -------
    Model : sklearn.[model]
        A ML model
    
    """
    if model_type == "RF":
        rf = RandomForestClassifier(n_estimators=10, random_state=1)
        rf.fit(X_train, y_train)
        model = rf
    elif model_type == "DT":
        tree = DecisionTreeClassifier(max_depth=3, random_state=1)
        tree.fit(X_train, y_train)
        model = tree
    elif model_type == "LR":
        logreg = LogisticRegression(max_iter=10000).fit(X_train, y_train)
        logreg.fit(X_train, y_train)
        model = logreg
    elif model_type == "KNN":
        knn = KNeighborsClassifier(n_neighbors=9)
        knn.fit(X_train, y_train)
        model = knn
    elif model_type == "GB":
        gb = GradientBoostingClassifier(random_state=1)
        gb.fit(X_train, y_train)
        model = gb
    elif model_type == "SVM":
        svc = SVC(C=1000)
        svc.fit(X_train, y_train)
        model = svc
    return model

def plotAccuracies(x, base_accuracy, model_type, train_test, noise):
    """
    This function plots the accuracies of the models.
        
    Parameters:
    ----------
    x : [float]
        Model accuracies
    base_accuracy : [float]
        Base model accuracies
    model_type : str
        Type of model
    train_test : str
        Identicates if accuracies are on training or test set
    noise : str
        Type of noise added

    Returns
    -------
    None.
    
    """
    if np.amax(x) > np.amax(base_accuracy):
        y = np.amax(x)
    else:
        y = np.amax(base_accuracy)
    fig, axis = plt.subplots(figsize =(10, 5))
    axis.hist(x, bins = np.linspace(np.amin(x), y, 20))
    plt.axvline(y, color='k', linestyle='dashed', linewidth=1)
    plt.title(model_type + " " + train_test + " Set Accuracy under " + noise + " Noise")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Number of Samples")
    plt.show()

def plotConfusionMatrix(y_true, y_pred, title):
    """
    This function takes the true and predicted labels for the model and plots
    a confusion matrix using them.
    
    Parameters:
    ----------
    y_true : [[int]]
        True labels
    y_pred : [[int]]
        Predicted labels
    title : str
        Title for the confusion matrix

    Returns
    -------
    None.
    
    """
    sn.heatmap(confusion_matrix(y_true, y_pred), annot=True, 
               xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.title("Test Set " + title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plotFeatureImportancesDiabetes(model, diabetes_features, title):
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
    plt.title(title)
    plt.show()

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

def addLaplaceNoiseRNM(dataSet, mu, scale, features):
    """
    This function takes a pandas dataframe and introduces random noise to it from
    the Laplace distribution. The parameters of said distribution are defined
    in the function call. This noise is added in a random noise max manner
    where the max is taken from the value in the data and the laplace noise.
    
    Note: This doesn't work
    
    Parameters
    ----------
    dataSet : Pandas.Dataframe
        The dataset that you wish to add the noise to.
    mu : float
        The mean for the laplace distribution.
    scale : float
        The standard deviation.
    features : [str]
        Features to add noise to.

    Returns
    -------
    privateDataSet : Pandas.Dataframe
        A copy of the dataSet with laplace random noise added.

    """
    privateDataSet = dataSet.copy()
    #Generate the noise to be added to the data
    noiseArray = [np.random.laplace(mu, scale) for x in range(dataSet.size)]
    
    #Normalise the noise to have the same bounds as the data
    normalNoiseArray = [(x - np.min(noiseArray))/(np.max(noiseArray) - np.min(noiseArray)) for x in noiseArray]
    
    noiseIndex = 0
    
    for feature in features:
        for index in range(dataSet[feature].size):
            privateDataSet[feature][index] = max(dataSet[feature][index], normalNoiseArray[noiseIndex])
            noiseIndex += 1
    return privateDataSet

def addLaplaceNoiseCell(dataSet, mu, scale, features):
    """
    This function takes a pandas dataframe and adds random noise to it from
    the Laplace distribution. The parameters of said distribution are defined
    in the function call.
    
    Parameters
    ----------
    dataSet : Pandas.Dataframe
        The dataset that you wish to add the noise to.
    mu : float
        The mean for the laplace distribution.
    scale : float
        The standard deviation.
    features : [str]
        Features to add noise to.

    Returns
    -------
    privateDataSet : Pandas.Dataframe
        A copy of the dataSet with laplace random noise added.

    """
    privateDataSet = dataSet.copy()
    for feature in features:
        for index in range(dataSet[feature].size):
            privateDataSet[feature][index] += np.random.laplace(mu, scale)
    return privateDataSet

def addLaplaceNoiseRow(dataSet, mu, scale, features):
    """
    This function takes a pandas dataframe and adds random noise to it from
    the Laplace distribution. The parameters of said distribution are defined
    in the function call. The same noise is added to each row.
    
    Note: This doesn't work, it gives very strange results
    
    Parameters
    ----------
    dataSet : Pandas.Dataframe
        The dataset that you wish to add the noise to.
    mu : float
        The mean for the laplace distribution.
    scale : float
        The standard deviation.
    features : [str]
        Features to add noise to.

    Returns
    -------
    privateDataSet : Pandas.Dataframe
        A copy of the dataSet with laplace random noise added.

    """
    privateDataSet = dataSet.copy()
    for feature in features:
        noise = np.random.laplace(mu, scale)
        for index in range(dataSet[feature].size):
            privateDataSet[feature][index] += noise
    return privateDataSet
    
def addGaussianNoiseCell(dataSet, mu, scale, features):
    """
    This function takes a pandas dataframe and adds random noise to it from
    the Gaussian distribution. The parameters of said distribution are defined
    in the function call. Additionally, the random noise is added individually
    each cell in the dataframe.
    
    Parameters
    ----------
    dataSet : Pandas.Dataframe
        The dataset that you wish to add the noise to.
    mu : float
        The mean for the laplace distribution.
    scale : float
        The standard deviation.
    features : [str]
        Features to add noise to.

    Returns
    -------
    privateDataSet : Pandas.Dataframe
        A copy of the dataSet with gaussian random noise added.

    """
    privateDataSet = dataSet.copy()
    for feature in features:
        for index in range(dataSet[feature].size):
            privateDataSet[feature][index] += np.random.normal(mu, scale)
    return privateDataSet
    
def addGaussianNoiseRow(dataSet, mu, scale, features):
    """
    This function takes a pandas dataframe and adds random noise to it from
    the Gaussian distribution. The parameters of said distribution are defined
    in the function call. Additionally, the random noise is added to each row
    so the noise to each feature in that row is the same.
    
    Note: This doesn't work, it gives very strange results
    
    Parameters
    ----------
    dataSet : Pandas.Dataframe
        The dataset that you wish to add the noise to.
    mu : float
        The mean for the laplace distribution.
    scale : float
        The standard deviation.
    features : [str]
        Features to add noise to.

    Returns
    -------
    privateDataSet : Pandas.Dataframe
        A copy of the dataSet with gaussian random noise added.

    """
    privateDataSet = dataSet.copy()
    for feature in features:
        noise = np.random.normal(mu, scale)
        for index in range(dataSet[feature].size):
            privateDataSet[feature][index] += noise
    return privateDataSet

def addStaircaseNoise(dataSet, epsilon, delta, gamma, features):
    """
    This function takes a pandas dataframe and adds random noise to it from
    the Staircase distribution. The parameters of said distribution are defined
    in the function call.
        
    Parameters
    ----------
    dataSet : Pandas.Dataframe
        The dataset that you wish to add the noise to.
    mu : float
        The mean for the laplace distribution.
    scale : float
        The standard deviation.
    features : [str]
        Features to add noise to.

    Returns
    -------
    privateDataSet : Pandas.Dataframe
        A copy of the dataSet with gaussian random noise added.

    """
    privateDataSet = dataSet.copy()
    b = np.exp(-epsilon)
    probs = [probability_to_occur_at(z, b) for z in range(100)]
    for feature in features:
        for index in range(dataSet[feature].size):
            privateDataSet[feature][index] += staircaseRV(epsilon, delta, gamma, probs, b)
    return privateDataSet
    
def probability_to_occur_at(attempt, b):
    """
    This function is used in the staircase random noise generation.
    It is effectively the probabilities from a binomial distribution.

    Parameters
    ----------
    attempt : int
        Number of the attempt.
    b : float
        Chance of success.

    Returns
    -------
    float
        Prob of success.

    """
    return (1-b)*b**(attempt) 

def geometricRV(b, probs):
    """
    This is a function I wrote to implement a geoemtric r.v. that was different
    to the number one.

    Parameters
    ----------
    b : float
        Chance of success.
    probs : list[float]
        Probabilities for the distribution    
    
    Returns
    -------
    G : int
        Random variable.

    """
    num = random.random()
    rollingSum = 0
    G = -1
    for prob in probs:
        rollingSum += prob
        if num <= rollingSum:
            G = probs.index(prob)
            break
    if G == -1:
        G = len(probs)
    return G

def staircaseRV(epsilon, delta, gamma, probs, b):
    """
    This function outputs a random variable from the staircase distribution.
    
    Parameters
    ----------
    epsilon : float
        The mean for the laplace distribution.
    sensitivity : float
        The standard deviation.
    gamma : float
        float in the range [0,1]

    Returns
    -------
    X : float
        Sample from staircase distribution

    """
    
    #First generate a r.v. S with 1/2 chance of being 1 and 1/2 chance
    #of being -1
    num = random.random()
    if num <= 0.5:
        S = -1
    else:
        S = 1
    
    #Second generate G    
    G = geometricRV(b, probs)
    
    #Thrid generate U uniformly distributioned between [0,1]
    U = random.uniform(0, 1)
    
    #Fourth generate B
    #Pr[B = 0] = gamma/(gamma + (1 - gamma)*b)
    #Pr[B = 1] = (1 - gamma)*b/(gamma + (1 - gamma)*b)
    num = random.random()
    if num <= gamma/(gamma + (1 - gamma)*b):
        B = 0
    else:
        B = 1
    
    X = S*((1 - B)*((G + gamma*U)*delta) + B*((G + gamma + (1 - gamma)*U)*delta))
    
    return X