#General Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Model Imports
from sklearn.model_selection import KFold

#Function Imports
from Data_Functions import addStaircaseNoise, generateModel, loadDiabetesData, addLaplaceNoise, addGaussianNoise, plotFeatureImportancesDiabetes, plotConfusionMatrix, plotAccuracies

def generateAccPrivacyGraph(model_type, trials, diabetes_features=[]):
    """
    """
    np_training_results = []
    np_testing_results = []
    l_training_results = []
    l_testing_results = []
    g_training_results = []
    g_testing_results = []
    s_training_results = []
    s_testing_results = []
    sensitivity = 1
    epsilons = np.linspace(0.01, 10, 50)
    for epsilon in epsilons:
        np_tr, l_tr, g_tr, s_tr, np_ts, l_ts, g_ts, s_ts = runModel(model_type, trials, sensitivity, epsilon, diabetes_features)
        np_training_results.append(np_tr)
        np_testing_results.append(np_ts)
        l_training_results.append(l_tr)
        l_testing_results.append(l_ts)
        g_training_results.append(g_tr)
        g_testing_results.append(g_ts)
        s_training_results.append(s_tr)
        s_testing_results.append(s_ts)
    
    plt.plot(np_training_results, epsilons, label="Non-Private")
    plt.plot(l_training_results, epsilons, label="Laplace Private")
    plt.plot(g_training_results, epsilons, label="Gaussian Private")
    plt.plot(s_training_results, epsilons, label="Staircase Private")
    plt.title("Training accuracies with different privacy levels")
    plt.xlabel("Accuracy")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.show()
    
    plt.plot(np_testing_results, epsilons, label="Non-Private")
    plt.plot(l_testing_results, epsilons, label="Laplace Private")
    plt.plot(g_testing_results, epsilons, label="Gaussian Private")
    plt.plot(s_testing_results, epsilons, label="Staircase Private")
    plt.title("Testing accuracies with different privacy levels")
    plt.xlabel("Accuracy")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.show()
    

def runModel(model_type, trials, sensitivity=1, epsilon=0.1, diabetes_features=[], gen_figures=False, gen_print=False):
    """
    This function should run whatever model is put in the model parameter.
    It should also run for the number of trials given.

    Parameters
    ----------
    model : str
        String representing the model to run.
    trials : int
        Number of trials to run for the noisy models.
    delta : int
        Proporational to the amount of noise added
    epsilon : float
        Inversely proporational to the amount of noise added
        
    Returns
    -------
    None.

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
        laplacePrivateDataset = addLaplaceNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_features)
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
        gaussianPrivateDataset = addGaussianNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, diabetes_features)
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
        staircasePrivateDataset = addStaircaseNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale, 0.5, diabetes_features)
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