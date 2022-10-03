import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import seaborn as sns

from Data_Functions import loadDiabetesData, addLaplaceNoise, addGaussianNoise, plotConfusionMatrix

sns.set_style('darkgrid')
#Load in the data
diabetes = loadDiabetesData()

#Number of trials to run
trials = 2

# Non-private LR Neighbours
kf = KFold(n_splits=5)
np_training_acc = []
np_testing_acc = []

for train, test in kf.split(diabetes):
    X_train, X_test, y_train, y_test = (diabetes.loc[train, diabetes.columns != 'Outcome'], 
                                        diabetes.loc[test, diabetes.columns != 'Outcome'], 
                                        diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                        diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())

    logreg = LogisticRegression(max_iter=10000).fit(X_train, y_train)
    np_training_acc.append(logreg.score(X_train, y_train))
    np_testing_acc.append(logreg.score(X_test, y_test))

print('Accuracy of non-private LR classifier on training set: {:.2f}'.format(np.mean(np_training_acc)))
print('Accuracy of non-private LR classifier on test set: {:.2f}'.format(np.mean(np_testing_acc)))

plotConfusionMatrix(y_test, logreg.predict(X_test), "Non-Private LR Confusion Matrix")

    
# Private Laplace Logistic Regression Model
# Generatet the private data set
delta = 1
epsilon = 0.1
scale = delta/epsilon

l_training_acc = []
l_testing_acc = []

for i in range(trials):
    laplacePrivateDataset = addLaplaceNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale)
    training_acc = []
    testing_acc = []
    
    for train, test in kf.split(diabetes):
        X_train, X_test, y_train, y_test = (laplacePrivateDataset.loc[train, laplacePrivateDataset.columns != 'Outcome'], 
                                            laplacePrivateDataset.loc[test, laplacePrivateDataset.columns != 'Outcome'], 
                                            diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                            diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
    
        logreg = LogisticRegression(max_iter=10000).fit(X_train, y_train)
        logreg.fit(X_train, y_train)
        training_acc.append(logreg.score(X_train, y_train))
        testing_acc.append(logreg.score(X_test, y_test))

    l_training_acc.append(np.mean(training_acc))
    l_testing_acc.append(np.mean(testing_acc))
    
print('Accuracy of laplace private LR classifier on training set: {:.2f}'.format(np.mean(l_training_acc)))
print('Accuracy of laplace private LR classifier on test set: {:.2f}'.format(np.mean(l_testing_acc)))

fig, axis = plt.subplots(figsize =(10, 5))
plt.hist(l_training_acc, bins = np.linspace(np.amin(l_training_acc), np.amax(np_training_acc), 20))
plt.axvline(np.mean(np_training_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("LR Training Set Accuracy under Laplace Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(l_testing_acc, bins = np.linspace(np.amin(l_testing_acc), np.amax(np_testing_acc), 20))
plt.axvline(np.mean(np_testing_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("LR Testing Set Accuracy under Laplace Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

plotConfusionMatrix(y_test, logreg.predict(X_test), "Laplace Private LR Confusion Matrix")

g_training_acc = []
g_testing_acc = []

# Private Gaussian Logistic Regression Model
for i in range(trials):
    gaussianPrivateDataset = addGaussianNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale)
    training_acc = []
    testing_acc = []
    
    for train, test in kf.split(diabetes):
        X_train, X_test, y_train, y_test = (gaussianPrivateDataset.loc[train, gaussianPrivateDataset.columns != 'Outcome'], 
                                            gaussianPrivateDataset.loc[test, gaussianPrivateDataset.columns != 'Outcome'], 
                                            diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                            diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
        
        logreg = LogisticRegression(max_iter=10000).fit(X_train, y_train)
        logreg.fit(X_train, y_train)
        training_acc.append(logreg.score(X_train, y_train))
        testing_acc.append(logreg.score(X_test, y_test))

    g_training_acc.append(np.mean(training_acc))
    g_testing_acc.append(np.mean(testing_acc))
print('Accuracy of gaussian private LR classifier on training set: {:.2f}'.format(np.mean(g_training_acc)))
print('Accuracy of gaussian private LR classifier on test set: {:.2f}'.format(np.mean(g_testing_acc)))

fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(g_training_acc, bins = np.linspace(np.amin(g_training_acc), np.amax(np_training_acc), 20))
plt.axvline(np.mean(np_training_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("LR Training Set Accuracy under Gaussian Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(g_testing_acc, bins = np.linspace(np.amin(g_testing_acc), np.amax(np_testing_acc), 20))
plt.axvline(np.mean(np_testing_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("LR Testing Set Accuracy under Gaussian Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

plotConfusionMatrix(y_test, logreg.predict(X_test), "Gaussian Private LR Confusion Matrix")