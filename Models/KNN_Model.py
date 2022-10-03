import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import seaborn as sns

from Data_Functions import loadDiabetesData, addLaplaceNoise, addGaussianNoise, plotConfusionMatrix

sns.set_style('darkgrid')
#Load in the data
diabetes = loadDiabetesData()

#Number of trials to run
trials = 2

# Non-private K-Nearest Neighbours
kf = KFold(n_splits=5)
np_training_acc = []
np_testing_acc = []

for train, test in kf.split(diabetes):
    X_train, X_test, y_train, y_test = (diabetes.loc[train, diabetes.columns != 'Outcome'], 
                                        diabetes.loc[test, diabetes.columns != 'Outcome'], 
                                        diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                        diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    np_training_acc.append(knn.score(X_train, y_train))
    np_testing_acc.append(knn.score(X_test, y_test))

print('Accuracy of non-private KNN classifier on training set: {:.2f}'.format(np.mean(np_training_acc)))
print('Accuracy of non-private KNN classifier on test set: {:.2f}'.format(np.mean(np_testing_acc)))

plotConfusionMatrix(y_test, knn.predict(X_test), "Non-Private KNN Confusion Matrix")

# Private K-Nearest Neighbours
# Generatet the private data set
delta = 1
epsilon = 0.1
scale = delta/epsilon

l_training_acc = []
l_testing_acc = []

# Generate a copy of the data set to make private
for i in range(trials):
    laplacePrivateDataset = addLaplaceNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale)
    training_acc = []
    testing_acc = []
    
    for train, test in kf.split(diabetes):
        X_train, X_test, y_train, y_test = (laplacePrivateDataset.loc[train, laplacePrivateDataset.columns != 'Outcome'], 
                                            laplacePrivateDataset.loc[test, laplacePrivateDataset.columns != 'Outcome'], 
                                            diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                            diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
    
        knn = KNeighborsClassifier(n_neighbors=9)
        knn.fit(X_train, y_train)
        training_acc.append(knn.score(X_train, y_train))
        testing_acc.append(knn.score(X_test, y_test))

    l_training_acc.append(np.mean(training_acc))
    l_testing_acc.append(np.mean(testing_acc))
print('Accuracy of laplace private KNN classifier on training set: {:.2f}'.format(np.mean(l_training_acc)))
print('Accuracy of laplace private KNN classifier on test set: {:.2f}'.format(np.mean(l_testing_acc)))

fig, axis = plt.subplots(figsize =(10, 5))
plt.hist(l_training_acc, bins = np.linspace(np.amin(l_training_acc), np.amax(np_training_acc), 20))
plt.axvline(np.mean(np_training_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("KNN Training Set Accuracy under Laplace Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(l_testing_acc, bins = np.linspace(np.amin(l_testing_acc), np.amax(np_testing_acc), 20))
plt.axvline(np.mean(np_testing_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("KNN Testing Set Accuracy under Laplace Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

plotConfusionMatrix(y_test, knn.predict(X_test), "Laplace Private KNN Confusion Matrix")

g_training_acc = []
g_testing_acc = []

# Generate a copy of the data set to make private
for i in range(trials):
    gaussianPrivateDataset = addGaussianNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale)
    training_acc = []
    testing_acc = []
    
    for train, test in kf.split(diabetes):
        X_train, X_test, y_train, y_test = (gaussianPrivateDataset.loc[train, gaussianPrivateDataset.columns != 'Outcome'], 
                                            gaussianPrivateDataset.loc[test, gaussianPrivateDataset.columns != 'Outcome'], 
                                            diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                            diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
    
        knn = KNeighborsClassifier(n_neighbors=9)
        knn.fit(X_train, y_train)
        training_acc.append(knn.score(X_train, y_train))
        testing_acc.append(knn.score(X_test, y_test))

    g_training_acc.append(np.mean(training_acc))
    g_testing_acc.append(np.mean(testing_acc))
print('Accuracy of gaussian private KNN classifier on training set: {:.2f}'.format(np.mean(g_training_acc)))
print('Accuracy of gaussian private KNN classifier on test set: {:.2f}'.format(np.mean(g_testing_acc)))

fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(g_training_acc, bins = np.linspace(np.amin(g_training_acc), np.amax(np_training_acc), 20))
plt.axvline(np.mean(np_training_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("KNN Training Set Accuracy under Gaussian Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(g_testing_acc, bins = np.linspace(np.amin(g_testing_acc), np.amax(np_testing_acc), 20))
plt.axvline(np.mean(np_testing_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("KNN Testing Set Accuracy under Gaussian Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

plotConfusionMatrix(y_test, knn.predict(X_test), "Gaussian Private KNN Confusion Matrix")
