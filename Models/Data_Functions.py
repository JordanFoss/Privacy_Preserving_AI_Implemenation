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

def addLaplaceNoise(dataSet, mu, scale, features):
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
    for feature in features:
        for index in range(dataSet[feature].size):
            privateDataSet[feature][index] += staircaseRV(epsilon, delta, gamma)
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

def staircaseRV(epsilon, sensitivity, gamma):
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
    privateDataSet : Pandas.Dataframe
        A copy of the dataSet with gaussian random noise added.

    """
    
    #First generate a r.v. S with 1/2 chance of being 1 and 1/2 chance
    #of being -1
    num = random.random()
    if num <= 0.5:
        S = -1
    else:
        S = 1
    
    #Second generate G
    b = np.exp(-epsilon)
    probs = [probability_to_occur_at(z, b) for z in range(100)]
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
    
    X = S*((1 - B)*((G + gamma*U)*sensitivity) + B*((G + gamma + (1 - gamma)*U)*sensitivity))
    
    return X


# These functions are for the Differentially private gradient descent model
# The thing to keep in mind with this is that noise is not added to the data
# noise is just added to the gradient as it searchs for the minimum loss
def gradient(theta, xi, yi):
    exponent = yi * (xi.dot(theta))
    return - (yi*xi) / (1+np.exp(exponent))

def L2_clip(v, b):
    norm = np.linalg.norm(v, ord=2)
    
    if norm > b:
        return b * (v / norm)
    else:
        return v

def avg_grad(theta, X, y):
    grads = [gradient(theta, xi, yi) for xi, yi in zip(X, y)]
    return np.mean(grads, axis=0)

def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)

def gaussian_mech(v, sensitivity, epsilon, delta):
    return v + np.random.normal(loc=0, scale=sensitivity * np.sqrt(2*np.log(1.25/delta)) / epsilon)

def gaussian_mech_vec(v, sensitivity, epsilon, delta):
    return v + np.random.normal(loc=0, scale=sensitivity * np.sqrt(2*np.log(1.25/delta)) / epsilon, size=len(v))

def gradient_sum(theta, X, y, b):
    gradients = [L2_clip(gradient(theta, x_i, y_i), b) for x_i, y_i in zip(X,y)]
        
    # sum query
    # L2 sensitivity is b (by clipping performed above)
    return np.sum(gradients, axis=0)

def noisy_gradient_descent(X_train, y_train, iterations, epsilon, delta):
    # Starts with a guess model of all zeros
    theta = np.zeros(X_train.shape[1])
    
    sensitivity = 5.0
    
    # A count of the number of training elements with some added noise
    # Compute a noisy count of the number of training examples (sensitivity 1)
    noisy_count = gaussian_mech(X_train.shape[0], 1, epsilon, delta)

    for i in range(iterations):
        # Add noise to the sum of the gradients based on its sensitivity
        grad_sum        = gradient_sum(theta, X_train, y_train, sensitivity)
        noisy_grad_sum  = gaussian_mech_vec(grad_sum, sensitivity, epsilon, delta)
        
        # Divide the noisy sum from (1) by the noisy count from (2)
        noisy_avg_grad  = noisy_grad_sum / noisy_count
        theta           = theta - noisy_avg_grad

    return theta
