#General Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from keras.layers import Dense,Dropout
from tensorflow import keras

#Model Imports
from sklearn.model_selection import KFold

#Function Imports
from Data_Functions import *

def plotDataBeforeAndAfterNoise(diabetes_feature=[]):
    """
    This function generates a plot of the dataset before noise is added and
    after. This is useful for gaining a visual understanding of what adding
    noise, and thus privacy is doing to the data.

    Parameters
    ----------
    diabetes_features : [str]
        List of features to add noise to. Defaults to all features to achieve
        epsilon-delta privacy
    

    Returns
    -------
    Graphs showing noise before and after noise is added

    """
    
    diabetes = loadDiabetesData()
    diabetes = diabetes.rename(columns={'DiabetesPedigreeFunction' : 'DPF'})
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_feature_labels = [x for i,x in enumerate(diabetes.columns) if i!=8]
        
    sensitivity = 1
    epsilon = 1
    
    #Standard Deviation for noise
    scale = sensitivity/epsilon
    diabetes.boxplot(column=diabetes_feature_labels, grid=False, rot=75, fontsize=10)
    plt.title("Original Dataset")
    plt.show()
    
    laplacePrivateDataset = addLaplaceNoiseCell(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_feature)
    
    scaler = MinMaxScaler()
    scaler.fit(laplacePrivateDataset)
    scaled = scaler.fit_transform(laplacePrivateDataset)
    laplacePrivateDataset = pd.DataFrame(scaled, columns=diabetes_feature_labels)
    
    laplacePrivateDataset.boxplot(column=diabetes_feature_labels, grid=False, rot=75, fontsize=10)
    plt.title("Laplace Private Dataset With Epsilon=" + str(epsilon) + " Privacy Budget")
    plt.show()
    
    gaussianPrivateDataset = addGaussianNoiseCell(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_feature)
    
    scaler.fit(gaussianPrivateDataset)
    scaled = scaler.fit_transform(gaussianPrivateDataset)
    gaussianPrivateDataset = pd.DataFrame(scaled, columns=diabetes_feature_labels)
    
    gaussianPrivateDataset.boxplot(column=diabetes_feature_labels, grid=False, rot=75, fontsize=10)
    plt.title("Gaussian Private Dataset With Epsilon=" + str(epsilon) + " Privacy Budget")
    plt.show()
    
    staircasePrivateDataset = addStaircaseNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 5, scale, 0.5, diabetes_feature)
    scaler.fit(staircasePrivateDataset)
    scaled = scaler.fit_transform(staircasePrivateDataset)
    staircasePrivateDataset = pd.DataFrame(scaled, columns=diabetes_feature_labels)
    
    staircasePrivateDataset.boxplot(column=diabetes_feature_labels, grid=False, rot=75, fontsize=10)
    plt.title("Staricase Private Dataset With Epsilon=" + str(epsilon) + " Privacy Budget")
    plt.show()

def generateDeepAccPrivacyGraph(noiseType, trials, diabetes_features=[]):
    """
    This function generates accuracy graphs for a model type with a given number
    of trials (since random noise is added multiple trials
    are taken to negate the varience). This results are then ploted to show 
    accuracy over a range of different epsilons.

    Parameters
    ----------
    model : str
        String representing the model to run.
    trials : int
        Number of trials to run for the noisy models.
    diabetes_features : [str]
        List of features using for feature importance analysis
        
    Returns
    -------
    Graphs showing accuracy for non-private and laplace noise
    
    """
    #Load in the data
    diabetes = loadDiabetesData()
    
    np.random.seed(5121999)
        
    model=keras.Sequential()
    model.add(Dense(15,input_dim=8, activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="SGD", metrics=['accuracy'])
    
    # Generate the Non-Private accuracies for comparsion
    np_tr, np_ts = runModelNonPrivate("RF", diabetes)
    
    dl_training_results = []
    dl_testing_results = []
    ds_training_results = []
    ds_testing_results = []
    sensitivity = 1
    epsilons = np.linspace(0.5, 10, 20)
    for epsilon in epsilons:
        dl_tr, dl_ts = runDeepModel(model, noiseType, trials, diabetes, sensitivity, epsilon)
        ds_tr, ds_ts = runDeepModel(model, "Staircase", trials, diabetes, sensitivity, epsilon)
        dl_training_results.append(dl_tr)
        dl_testing_results.append(dl_ts)
        ds_training_results.append(ds_tr)
        ds_testing_results.append(ds_ts)
    
    fig = plt.figure(figsize =(10, 7))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.errorbar(epsilons, [np.mean(x) for x in dl_training_results], yerr=[np.std(x) for x in dl_training_results],
                capsize=5, label="Laplace Private")
    ax.errorbar(epsilons, [np.mean(x) for x in ds_training_results], yerr=[np.std(x) for x in ds_training_results],
                capsize=5, label="Staircase Private")
    ax.plot(epsilons, [np_tr for x in range(len(epsilons))], label="Non-Private")
    plt.title("Deep Learning Training Accuracies (" + str(trials) + " trials)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.legend()
    plt.show()
    
    fig = plt.figure(figsize =(8, 5))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.errorbar(epsilons, [np.mean(x) for x in dl_testing_results], yerr=[np.std(x) for x in dl_testing_results],
                capsize=5, label="Laplace Private")
    ax.errorbar(epsilons, [np.mean(x) for x in ds_testing_results], yerr=[np.std(x) for x in ds_testing_results],
                capsize=5, label="Staircase Private")
    ax.plot(epsilons, [np_ts for x in range(len(epsilons))], label="Non-Private")
    plt.title("Deep Learning Testing Accuracies (" + str(trials) + " trials)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.legend()
    plt.show()

def generateAccPrivacyGraphLaplace(model_type, trials, diabetes_features=[]):
    """
    This function generates accuracy graphs for a model type with a given number
    of trials (since random noise is added multiple trials
    are taken to negate the varience). This results are then ploted to show 
    accuracy over a range of different epsilons.

    Parameters
    ----------
    model : str
        String representing the model to run.
    trials : int
        Number of trials to run for the noisy models.
    diabetes_features : [str]
        List of features using for feature importance analysis
        
    Returns
    -------
    Graphs showing accuracy for non-private and laplace noise
    
    """
    #Load in the data
    diabetes = loadDiabetesData()
    
    # Generate the Non-Private accuracies for comparsion
    np_tr, np_ts = runModelNonPrivate(model_type, diabetes)
    
    l_training_results = []
    l_testing_results = []
    sensitivity = 1
    epsilons = np.linspace(0.5, 10, 20)
    for epsilon in epsilons:
        l_tr, l_ts = runModelLaplace(model_type, trials, diabetes, sensitivity, epsilon)
        l_training_results.append(l_tr)
        l_testing_results.append(l_ts)
    
    fig = plt.figure(figsize =(10, 7))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.errorbar(epsilons, [np.mean(x) for x in l_training_results], yerr=[np.std(x) for x in l_training_results],
                capsize=5, label="Laplace Private")
    ax.plot(epsilons, [np_tr for x in range(len(epsilons))], label="Non-Private")
    plt.title(str(model_type) + " Training Accuracies (" + str(trials) + " trials)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.legend()
    plt.show()
    
    fig = plt.figure(figsize =(8, 5))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.errorbar(epsilons, [np.mean(x) for x in l_testing_results], yerr=[np.std(x) for x in l_testing_results],
                capsize=5, label="Laplace Private")
    ax.plot(epsilons, [np_ts for x in range(len(epsilons))], label="Non-Private")
    plt.title(str(model_type) + " Testing Accuracies (" + str(trials) + " trials)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.legend()
    plt.show()
    
    return l_training_results, l_testing_results


def generateAccPrivacyGraphStaircase(model_type, trials, diabetes_features=[]):
    """
    This function generates accuracy graphs for a model type with a given number
    of trials (since random noise is added multiple trials
    are taken to negate the varience). This results are then ploted to show 
    accuracy over a range of different epsilons.

    Parameters
    ----------
    model : str
        String representing the model to run.
    trials : int
        Number of trials to run for the noisy models.
    diabetes_features : [str]
        List of features using for feature importance analysis
        
    Returns
    -------
    Graphs showing accuracy for non-private and laplace noise
    
    """
    #Load in the data
    diabetes = loadDiabetesData()
    
    # Generate the Non-Private accuracies for comparsion
    np_tr, np_ts = runModelNonPrivate(model_type, diabetes)
    
    s_training_results = []
    s_testing_results = []
    sensitivity = 1
    epsilons = np.linspace(0.5, 10, 20)
    staircaseEpsilons = np.linspace(0.5, 3, 6)
    counter = 0
    for epsilon in epsilons:
        staircaseEpsilonTrainingResults = []
        staircaseEpsilonTestingResults = []
        for staircaseEpsilon in staircaseEpsilons:
            s_tr, s_ts = runModelStaircase(model_type, trials, diabetes, staircaseEpsilon, sensitivity, epsilon)
            staircaseEpsilonTrainingResults.append(s_tr)
            staircaseEpsilonTestingResults.append(s_ts)
        print(counter)
        counter += 1
        s_training_results.append(staircaseEpsilonTrainingResults)
        s_testing_results.append(staircaseEpsilonTestingResults)
    
    sTrainingFinalResults = []
    sTestingFinalResults = []
    for index in range(len(staircaseEpsilons)):
        sTrainingFinalResults.append([x[index] for x in s_training_results])
        sTestingFinalResults.append([x[index] for x in s_testing_results])
    
    fig = plt.figure(figsize =(10, 7))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    for index in range(len(staircaseEpsilons)):
        ax.errorbar(epsilons, [np.mean(x) for x in sTrainingFinalResults[index]], yerr=0,
                    capsize=5, label="Staircase Private "+str(staircaseEpsilons[index]))
    ax.plot(epsilons, [np_tr for x in range(len(epsilons))], label="Non-Private")
    plt.title(str(model_type) + " Training Accuracies (" + str(trials) + " trials)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.legend()
    plt.show()
    
    fig = plt.figure(figsize =(8, 5))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    for index in range(len(staircaseEpsilons)):
        ax.errorbar(epsilons, [np.mean(x) for x in sTestingFinalResults[index]], yerr=0,
                    capsize=5, label="Staircase Private"+str(staircaseEpsilons[index]))
    ax.plot(epsilons, [np_ts for x in range(len(epsilons))], label="Non-Private")
    plt.title(str(model_type) + " Testing Accuracies (" + str(trials) + " trials)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.legend()
    plt.show()

def generateAccPrivacyGraphAll(model_type, trials, diabetes_features=[]):
    """
    This function plots the accuracy for all the mechnaism types for a
    given model type and number of trials. This plot is a line plot that
    also shows the standard deviation for the different values of epsilon,
    as the noise is randomly added multiple trials are required to get an
    accurate measure for the accuracy of the model at a given epsilon.

    Parameters
    ----------
    model : str
        String representing the model to run.
    trials : int
        Number of trials to run for the noisy models.
    diabetes_features : [str]
        List of features using for feature importance analysis
        
    Returns
    -------
    Graphs showing accuracy for non-private, laplace, gaussian and staircase noise
    
    """
    #Load in the data
    diabetes = loadDiabetesData()
    
    np.random.seed(5121999)
        
    model=keras.Sequential()
    model.add(Dense(15,input_dim=8, activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="SGD", metrics=['accuracy'])
    
    # Generate the Non-Private accuracies for comparsion
    np_tr, np_ts = runModelNonPrivate(model_type, diabetes)
    
    # Generate the Non-Private accuracies for comparsion
    dnp_tr, dnp_ts = runModelDeepNonPrivate(model, diabetes)
    
    # ML results
    l_training_results = []
    l_testing_results = []
    g_training_results = []
    g_testing_results = []
    s_training_results = []
    s_testing_results = []
    
    #DL results
    dl_training_results = []
    dl_testing_results = []
    dg_training_results = []
    dg_testing_results = []
    ds_training_results = []
    ds_testing_results = []
    
    sensitivity = 1
    epsilons = np.linspace(0.5, 10, 10)
    for epsilon in epsilons:
        l_tr, l_ts = runModelLaplace(model_type, trials, diabetes, sensitivity, epsilon, diabetes_features)
        g_tr, g_ts = runModelGaussian(model_type, trials, diabetes, sensitivity, epsilon, diabetes_features)
        s_tr, s_ts = runModelStaircase(model_type, trials, diabetes, epsilon, sensitivity, diabetes_features)
        dl_tr, dl_ts = runDeepModel(model, "Laplace", trials, diabetes, sensitivity, epsilon)
        dg_tr, dg_ts = runDeepModel(model, "Gaussian", trials, diabetes, sensitivity, epsilon)
        ds_tr, ds_ts = runDeepModel(model, "Staircase", trials, diabetes, sensitivity, epsilon)
        l_training_results.append(l_tr)
        l_testing_results.append(l_ts)
        g_training_results.append(g_tr)
        g_testing_results.append(g_ts)
        s_training_results.append(s_tr)
        s_testing_results.append(s_ts)
        dl_training_results.append(dl_tr)
        dl_testing_results.append(dl_ts)
        dg_training_results.append(dg_tr)
        dg_testing_results.append(dg_ts)
        ds_training_results.append(ds_tr)
        ds_testing_results.append(ds_ts)
    
    fig = plt.figure(figsize =(10, 7))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.errorbar(epsilons, [np.mean(x) for x in l_training_results], yerr=[np.std(x) for x in l_training_results],
                capsize=5, label="Laplace Private")
    ax.errorbar(epsilons, [np.mean(x) for x in g_training_results], yerr=[np.std(x) for x in g_training_results],
                capsize=5, label="Gaussian Private")
    ax.errorbar(epsilons, [np.mean(x) for x in s_training_results], yerr=[np.std(x) for x in s_training_results],
                capsize=5, label="Staircase Private")
    
    ax.errorbar(epsilons, [np.mean(x) for x in dl_training_results], yerr=[np.std(x) for x in dl_training_results],
                capsize=5, label="Deep Laplace Private")
    ax.errorbar(epsilons, [np.mean(x) for x in dg_training_results], yerr=[np.std(x) for x in dg_training_results],
                capsize=5, label="Deep Gaussian Private")
    ax.errorbar(epsilons, [np.mean(x) for x in ds_training_results], yerr=[np.std(x) for x in ds_training_results],
                capsize=5, label="Deep Staircase Private")
    ax.plot(epsilons, [np_tr for x in range(len(epsilons))], label="Non-Private")
    ax.plot(epsilons, [dnp_tr for x in range(len(epsilons))], label="Deep Non-Private")
    plt.title(str(model_type) + " Training Accuracies (" + str(trials) + " trials)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.legend()
    plt.show()
    
    fig = plt.figure(figsize =(8, 5))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.errorbar(epsilons, [np.mean(x) for x in l_testing_results], yerr=[np.std(x) for x in l_testing_results],
                capsize=5, label="Laplace Private")
    ax.errorbar(epsilons, [np.mean(x) for x in g_testing_results], yerr=[np.std(x) for x in g_testing_results],
                capsize=5, label="Gaussian Private")
    ax.errorbar(epsilons, [np.mean(x) for x in s_testing_results], yerr=[np.std(x) for x in s_testing_results],
                capsize=5, label="Staircase Private")
    ax.errorbar(epsilons, [np.mean(x) for x in dl_testing_results], yerr=[np.std(x) for x in dl_testing_results],
                capsize=5, label="Deep Laplace Private")
    ax.errorbar(epsilons, [np.mean(x) for x in dg_testing_results], yerr=[np.std(x) for x in dg_testing_results],
                capsize=5, label="Deep aussian Private")
    ax.errorbar(epsilons, [np.mean(x) for x in ds_testing_results], yerr=[np.std(x) for x in ds_testing_results],
                capsize=5, label="Deep Staircase Private")
    ax.plot(epsilons, [np_ts for x in range(len(epsilons))], label="Non-Private")
    ax.plot(epsilons, [dnp_ts for x in range(len(epsilons))], label="Deep Non-Private")
    plt.title(str(model_type) + " Testing Accuracies (" + str(trials) + " trials)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epsilon")
    plt.legend()
    plt.show()
    
    
def runModelDeepNonPrivate(model, diabetes, diabetes_features=[]):    
    """
    This function trains a deep learning model of the given model type on the original dataset.

    Parameters
    ----------
    diabetes : pd.dataframe
        Original data to train on
    diabetes_features : [str], optional
        Features to add noise to. The default is [].

    Returns
    -------
    [float]
        Average training and testing accuracy fron the k-folds cross validation.

    """
    
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_feature_labels = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    if len(diabetes_features) == 0:
        diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    # Non-private RF Neighbours
    kf = KFold(n_splits=5)
    np_training_acc = []
    np_testing_acc = []
    
    for train, test in kf.split(diabetes):
        X_train, X_test, y_train, y_test = (diabetes.loc[train, diabetes.columns != 'Outcome'], 
                                            diabetes.loc[test, diabetes.columns != 'Outcome'], 
                                            diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                            diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=0, validation_data=(X_test, y_test))
        np_testing_acc.append(model.evaluate(X_test, y_test, verbose=0)[1])
        np_training_acc.append(model.evaluate(X_train, y_train, verbose=0)[1])
    return np.mean(np_training_acc), np.mean(np_testing_acc)
  
    
def runModelNonPrivate(model_type, diabetes, diabetes_features=[]):   
    """
    This function trains a model of the given model type on the original dataset.

    Parameters
    ----------
    model_type : str
        Type of model to use
    diabetes : pd.dataframe
        Original data to train on
    diabetes_features : [str], optional
        Features to add noise to. The default is [].

    Returns
    -------
    [float]
        Average training and testing accuracy fron the k-folds cross validation.

    """
    
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_feature_labels = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    if len(diabetes_features) == 0:
        diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    # Non-private RF Neighbours
    kf = KFold(n_splits=5)
    np_training_acc = []
    np_testing_acc = []
    
    for train, test in kf.split(diabetes):
        X_train, X_test, y_train, y_test = (diabetes.loc[train, diabetes.columns != 'Outcome'], 
                                            diabetes.loc[test, diabetes.columns != 'Outcome'], 
                                            diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                            diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        model = generateModel(model_type, X_train, y_train)
        np_training_acc.append(model.score(X_train, y_train))
        np_testing_acc.append(model.score(X_test, y_test))  
    return np.mean(np_training_acc), np.mean(np_testing_acc)

def runDeepModel(model, noiseType, trials, diabetes, sensitivity=1, epsilon=0.1, diabetes_features=[]):
    """
    This function trains a deep learning model on the diabete dataset with
    Laplace added noise for a given number of trials with a given level
    of privacy epsilon. 

    Parameters
    ----------
    model_type : str
        Type of model to use
    diabetes : pd.dataframe
        Original data to train on
    sensitivity : float, optional
        Sensitivtiy for the dataset. The default is 1.
    epsilon : flaot, optional
        Level of privacy for the dataset, lower implies higher privacy. The default is 0.1.
    diabetes_features : [str], optional
        Features to add noise to. The default is [].

    Returns
    -------
    overallTrainingAcc : [flaot]
        Training accuracies from all the trials.
    overallTestingAcc : [flaot]
        Testing accuracies from all the trials.

    """
    #Standard Deviation for noise
    scale = sensitivity/epsilon
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_feature_labels = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    if len(diabetes_features) == 0:
        diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    # Non-private RF Neighbours
    kf = KFold(n_splits=5)
    scaler = MinMaxScaler()
    
    overallTrainingAcc = []
    overallTestingAcc = []
    
    for i in range(trials):
        if noiseType == "Laplace":
            privateDataset = addLaplaceNoiseCell(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_features)
        elif noiseType == "Gaussian":
            privateDataset = addGaussianNoiseCell(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_features)
        elif noiseType == "Staircase":
            privateDataset = addStaircaseNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], epsilon, scale, 0.5, diabetes_features)
        scaler.fit(privateDataset)
        scaled = scaler.fit_transform(privateDataset)
        privateDataset = pd.DataFrame(scaled, columns=diabetes_feature_labels)
        training_acc = []
        testing_acc = []
        
        for train, test in kf.split(diabetes):
            X_train, X_test, y_train, y_test = (privateDataset.loc[train, privateDataset.columns != 'Outcome'], 
                                                privateDataset.loc[test, privateDataset.columns != 'Outcome'], 
                                                diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                                diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
            model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=0, validation_data=(X_test, y_test))
            testing_acc.append(model.evaluate(X_test, y_test, verbose=0)[1])
            training_acc.append(model.evaluate(X_train, y_train, verbose=0)[1])
    
        overallTrainingAcc.append(np.mean(training_acc))
        overallTestingAcc.append(np.mean(testing_acc))
    
    # Model is done being generated
    return overallTrainingAcc, overallTestingAcc

def runModelLaplace(model_type, trials, diabetes, sensitivity=1, epsilon=0.1, diabetes_features=[]):
    """
    This function trains a model of model type on the diabete dataset with
    Laplace added noise for a given number of trials with a given level
    of privacy epsilon. 

    Parameters
    ----------
    model_type : str
        Type of model to use
    diabetes : pd.dataframe
        Original data to train on
    sensitivity : float, optional
        Sensitivtiy for the dataset. The default is 1.
    epsilon : flaot, optional
        Level of privacy for the dataset, lower implies higher privacy. The default is 0.1.
    diabetes_features : [str], optional
        Features to add noise to. The default is [].

    Returns
    -------
    overallTrainingAcc : [flaot]
        Training accuracies from all the trials.
    overallTestingAcc : [flaot]
        Testing accuracies from all the trials.

    """
    #Standard Deviation for noise
    scale = sensitivity/epsilon
    
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_feature_labels = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    if len(diabetes_features) == 0:
        diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    # Non-private RF Neighbours
    kf = KFold(n_splits=5)
    scaler = MinMaxScaler()
    
    overallTrainingAcc = []
    overallTestingAcc = []
    
    for i in range(trials):
        laplacePrivateDataset = addLaplaceNoiseCell(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_features)
        scaler.fit(laplacePrivateDataset)
        scaled = scaler.fit_transform(laplacePrivateDataset)
        laplacePrivateDataset = pd.DataFrame(scaled, columns=diabetes_feature_labels)
        training_acc = []
        testing_acc = []
        
        for train, test in kf.split(diabetes):
            X_train, X_test, y_train, y_test = (laplacePrivateDataset.loc[train, laplacePrivateDataset.columns != 'Outcome'], 
                                                laplacePrivateDataset.loc[test, laplacePrivateDataset.columns != 'Outcome'], 
                                                diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                                diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        
            model = generateModel(model_type, X_train, y_train)
            training_acc.append(model.score(X_train, y_train))
            testing_acc.append(model.score(X_test, y_test))
    
        overallTrainingAcc.append(np.mean(training_acc))
        overallTestingAcc.append(np.mean(testing_acc))
    
    # Model is done being generated
    return overallTrainingAcc, overallTestingAcc


def runModelGaussian(model_type, trials, diabetes, sensitivity=1, epsilon=0.1, diabetes_features=[]):
    """
    This function trains a model of model type on the diabete dataset with
    Gaussian added noise for a given number of trials with a given level
    of privacy epsilon. 

    Parameters
    ----------
    model_type : str
        Type of model to use
    diabetes : pd.dataframe
        Original data to train on
    sensitivity : float, optional
        Sensitivtiy for the dataset. The default is 1.
    epsilon : flaot, optional
        Level of privacy for the dataset, lower implies higher privacy. The default is 0.1.
    diabetes_features : [str], optional
        Features to add noise to. The default is [].

    Returns
    -------
    overallTrainingAcc : [flaot]
        Training accuracies from all the trials.
    overallTestingAcc : [flaot]
        Testing accuracies from all the trials.

    """
    #Standard Deviation for noise
    scale = sensitivity/epsilon
    
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_feature_labels = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    if len(diabetes_features) == 0:
        diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    # Non-private RF Neighbours
    kf = KFold(n_splits=5)
    scaler = MinMaxScaler()
    
    overallTrainingAcc = []
    overallTestingAcc = []
    
    for i in range(trials):
        gaussianPrivateDataset = addGaussianNoiseCell(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_features)
        scaler.fit(gaussianPrivateDataset)
        scaled = scaler.fit_transform(gaussianPrivateDataset)
        gaussianPrivateDataset = pd.DataFrame(scaled, columns=diabetes_feature_labels)
        training_acc = []
        testing_acc = []
        
        for train, test in kf.split(diabetes):
            X_train, X_test, y_train, y_test = (gaussianPrivateDataset.loc[train, gaussianPrivateDataset.columns != 'Outcome'], 
                                                gaussianPrivateDataset.loc[test, gaussianPrivateDataset.columns != 'Outcome'], 
                                                diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                                diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        
            model = generateModel(model_type, X_train, y_train)
            training_acc.append(model.score(X_train, y_train))
            testing_acc.append(model.score(X_test, y_test))
    
        overallTrainingAcc.append(np.mean(training_acc))
        overallTestingAcc.append(np.mean(testing_acc))
    
    # Model is done being generated
    return overallTrainingAcc, overallTestingAcc

def runModelStaircase(model_type, trials, diabetes, epsilon, sensitivity=1, diabetes_features=[]):
    """
    This function trains a model of model type on the diabete dataset with
    Staircase added noise for a given number of trials with a given level
    of privacy epsilon. 

    Parameters
    ----------
    model_type : str
        Type of model to use
    diabetes : pd.dataframe
        Original data to train on
    sensitivity : float, optional
        Sensitivtiy for the dataset. The default is 1.
    epsilon : flaot, optional
        Level of privacy for the dataset, lower implies higher privacy.
    diabetes_features : [str], optional
        Features to add noise to. The default is [].

    Returns
    -------
    overallTrainingAcc : [flaot]
        Training accuracies from all the trials.
    overallTestingAcc : [flaot]
        Testing accuracies from all the trials.

    """
    #Standard Deviation for noise
    scale = sensitivity/epsilon
    
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_feature_labels = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    if len(diabetes_features) == 0:
        diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    # Non-private RF Neighbours
    kf = KFold(n_splits=5)
    scaler = MinMaxScaler()
    
    overallTrainingAcc = []
    overallTestingAcc = []
    
    for i in range(trials):
        #Note that increaing the epsilon value provides better results in the model
        staircasePrivateDataset = addStaircaseNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], epsilon, scale, 0.5, diabetes_features)
        scaler.fit(staircasePrivateDataset)
        scaled = scaler.fit_transform(staircasePrivateDataset)
        staircasePrivateDataset = pd.DataFrame(scaled, columns=diabetes_feature_labels)
        training_acc = []
        testing_acc = []
        
        for train, test in kf.split(diabetes):
            X_train, X_test, y_train, y_test = (staircasePrivateDataset.loc[train, staircasePrivateDataset.columns != 'Outcome'], 
                                                staircasePrivateDataset.loc[test, staircasePrivateDataset.columns != 'Outcome'], 
                                                diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                                diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        
            model = generateModel(model_type, X_train, y_train)
            training_acc.append(model.score(X_train, y_train))
            testing_acc.append(model.score(X_test, y_test))
    
        overallTrainingAcc.append(np.mean(training_acc))
        overallTestingAcc.append(np.mean(testing_acc))
    
    # Model is done being generated
    return overallTrainingAcc, overallTestingAcc

def runModelAll(model_type, trials, sensitivity=1, epsilon=0.1, diabetes_features=[], gen_figures=False, gen_print=False):
    """
    This function should run whatever model is put in the model parameter.
    It should also run for the number of trials given.
    
    Note: This code is outdated and generateAccPrivacyGraphAll now does much 
    tha same purpose. I'm keeping this code incase any aspects are useful.

    Parameters
    ----------
    model : str
        String representing the model to run.
    trials : int
        Number of trials to run for the noisy models.
    diabetes_features : [str]
        List of features using for feature importance analysis
    delta : int
        Proporational to the amount of noise added
    epsilon : float
        Inversely proporational to the amount of noise added
        
    Returns
    -------
    Training and test mean accuracy for non-private, laplace, gaussian and staircase noise.

    """
    
    feature_flag = False
    if model_type == "RF" or model_type == "DT" or model_type == "GB":
        feature_flag = True
    
    sns.set_style('darkgrid')
    #Load in the data
    diabetes = loadDiabetesData()    

    #Standard Deviation for noise
    scale = sensitivity/epsilon
    
    
    # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_feature_labels = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    if len(diabetes_features) == 0:
        diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
    
    # Non-private RF Neighbours
    kf = KFold(n_splits=5)
    np_training_acc = []
    np_testing_acc = []
    np_feature_importance = []
    np_mean_feature_importance = []
    
    for train, test in kf.split(diabetes):
        X_train, X_test, y_train, y_test = (diabetes.loc[train, diabetes.columns != 'Outcome'], 
                                            diabetes.loc[test, diabetes.columns != 'Outcome'], 
                                            diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                            diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        model = generateModel(model_type, X_train, y_train)
        np_training_acc.append(model.score(X_train, y_train))
        np_testing_acc.append(model.score(X_test, y_test))

        if feature_flag:
            np_feature_importance.append(model.feature_importances_)
    if gen_print:
        print('Accuracy of non-private ' + model_type +' classifier on training set: {:.2f}'.format(np.mean(np_training_acc)))
        print('Accuracy of non-private ' + model_type +' classifier on test set: {:.2f}'.format(np.mean(np_testing_acc)))
    if gen_figures:
        plotConfusionMatrix(y_test, model.predict(X_test), "Non-Private " + model_type + " Confusion Matrix")
    
    if feature_flag and gen_figures:
        for index in range(len(diabetes_features)):
            np_mean_feature_importance.append(np.mean([x[index] for x in np_feature_importance]))
        plotFeatureImportancesDiabetes(model, diabetes_feature_labels, "Non-Private " + model_type)
        plt.savefig('feature_importance')
    
    l_training_acc = []
    l_testing_acc = []
    l_feature_importance = []
    l_mean_feature_importance = []
    
    for i in range(trials):
        laplacePrivateDataset = addLaplaceNoiseCell(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_features)
        training_acc = []
        testing_acc = []
        
        for train, test in kf.split(diabetes):
            X_train, X_test, y_train, y_test = (laplacePrivateDataset.loc[train, laplacePrivateDataset.columns != 'Outcome'], 
                                                laplacePrivateDataset.loc[test, laplacePrivateDataset.columns != 'Outcome'], 
                                                diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                                diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        
            model = generateModel(model_type, X_train, y_train)
            training_acc.append(model.score(X_train, y_train))
            testing_acc.append(model.score(X_test, y_test))
            if feature_flag:
                l_feature_importance.append(model.feature_importances_)
    
        l_training_acc.append(np.mean(training_acc))
        l_testing_acc.append(np.mean(testing_acc))
        
    if gen_print:
        print('Accuracy of laplace private ' + model_type +' classifier on training set: {:.2f}'.format(np.mean(l_training_acc)))
        print('Accuracy of laplace private ' + model_type +' classifier on test set: {:.2f}'.format(np.mean(l_testing_acc)))
    
    if gen_figures:
        plotAccuracies(l_training_acc, np_training_acc, model_type, "Training", "Laplace")
        plotAccuracies(l_testing_acc, np_testing_acc, model_type, "Testing", "Laplace")
        plotConfusionMatrix(y_test, model.predict(X_test), "Laplace Private " + model_type + " Confusion Matrix")
    
    if feature_flag and gen_figures:
        for index in range(len(diabetes_features)):
            l_mean_feature_importance.append(np.mean([x[index] for x in l_feature_importance]))
        plotFeatureImportancesDiabetes(model, diabetes_feature_labels, "Laplace Private " + model_type)
        plt.savefig('feature_importance')
    
    
    g_training_acc = []
    g_testing_acc = []
    g_feature_importance = []
    g_mean_feature_importance = []
    
    # Private Gaussian Logistic Regression Model
    for i in range(trials):
        # Generate a copy of the data set to make private
        gaussianPrivateDataset = addGaussianNoiseCell(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_features)
        training_acc = []
        testing_acc = []
        
        for train, test in kf.split(diabetes):
            X_train, X_test, y_train, y_test = (gaussianPrivateDataset.loc[train, gaussianPrivateDataset.columns != 'Outcome'], 
                                                gaussianPrivateDataset.loc[test, gaussianPrivateDataset.columns != 'Outcome'], 
                                                diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                                diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        
            model = generateModel(model_type, X_train, y_train)
            training_acc.append(model.score(X_train, y_train))
            testing_acc.append(model.score(X_test, y_test))
            if feature_flag:
                g_feature_importance.append(model.feature_importances_)
    
        g_training_acc.append(np.mean(training_acc))
        g_testing_acc.append(np.mean(testing_acc))
       
    if gen_print:
        print('Accuracy of gaussian private ' + model_type +' classifier on training set: {:.2f}'.format(np.mean(g_training_acc)))
        print('Accuracy of gaussian private ' + model_type +' classifier on test set: {:.2f}'.format(np.mean(g_testing_acc)))
        
    if gen_figures:
        plotAccuracies(g_training_acc, np_training_acc, model_type, "Training", "Gaussian")
        plotAccuracies(g_testing_acc, np_testing_acc, model_type, "Testing", "Gaussian")
        plotConfusionMatrix(y_test, model.predict(X_test), "Gaussian Private " + model_type + " Confusion Matrix")
    
    if feature_flag and gen_figures:
        for index in range(len(diabetes_features)):
            g_mean_feature_importance.append(np.mean([x[index] for x in g_feature_importance]))
        plotFeatureImportancesDiabetes(model, diabetes_feature_labels, "Gaussian Private " + model_type)
        plt.savefig('feature_importance')   
        
    s_training_acc = []
    s_testing_acc = []
    s_feature_importance = []
    s_mean_feature_importance = []
    
    # Private Staircase Logistic Regression Model
    for i in range(trials):
        # Generate a copy of the data set to make private
        staircasePrivateDataset = addStaircaseNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0.5, scale, 0.5, diabetes_features)
        training_acc = []
        testing_acc = []
        
        for train, test in kf.split(diabetes):
            X_train, X_test, y_train, y_test = (staircasePrivateDataset.loc[train, staircasePrivateDataset.columns != 'Outcome'], 
                                                staircasePrivateDataset.loc[test, staircasePrivateDataset.columns != 'Outcome'], 
                                                diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                                diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        
            model = generateModel(model_type, X_train, y_train)
            training_acc.append(model.score(X_train, y_train))
            testing_acc.append(model.score(X_test, y_test))
            if feature_flag:
                s_feature_importance.append(model.feature_importances_)
    
        s_training_acc.append(np.mean(training_acc))
        s_testing_acc.append(np.mean(testing_acc))
        
    if gen_print:
        print('Accuracy of staircase private ' + model_type +' classifier on training set: {:.2f}'.format(np.mean(s_training_acc)))
        print('Accuracy of staircase private ' + model_type +' classifier on test set: {:.2f}'.format(np.mean(s_testing_acc)))
        
    if gen_figures:
        plotAccuracies(s_training_acc, np_training_acc, model_type, "Training", "Staircase")
        plotAccuracies(s_testing_acc, np_testing_acc, model_type, "Testing", "Staircase")
        plotConfusionMatrix(y_test, model.predict(X_test), "Staircase Private " + model_type + " Confusion Matrix")
    
    if feature_flag and gen_figures:
        for index in range(len(diabetes_features)):
            s_mean_feature_importance.append(np.mean([x[index] for x in s_feature_importance]))
        plotFeatureImportancesDiabetes(model, diabetes_feature_labels, "Staircase Private " + model_type)
        plt.savefig('feature_importance')
    
    return np.mean(np_training_acc), np.mean(l_training_acc), np.mean(g_training_acc), np.mean(s_training_acc), np.mean(np_testing_acc), np.mean(l_testing_acc), np.mean(g_testing_acc), np.mean(s_testing_acc)