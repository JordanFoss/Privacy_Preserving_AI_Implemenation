import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#Load in the data
diabetes = pd.read_csv('/home/jordan/Documents/UniversityStuff/2022/Semester2/ENGG4812/Privacy_Preserving_AI_Implemenation/Data_Sets/diabetes.csv')
pd.set_option('mode.chained_assignment', None)


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


# Non-private K-Nearest Neighbours
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 15
neighbors_settings = range(1, 20)

for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))


# Private K-Nearest Neighbours

# Generatet the private data set
delta = 1
epsilon = 0.1
scale = delta/epsilon

# Generate a copy of the data set to make private
laplacePrivateDataset = addLaplaceNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale)

# Private K-Nearest Neighbours
X_train, X_test, y_train, y_test = train_test_split(laplacePrivateDataset.loc[:, laplacePrivateDataset.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

training_accuracy = []
test_accuracy = []

for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of laplace private K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of laplace private K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

# Generate a copy of the data set to make private
gaussianPrivateDataset = addGaussianNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale)

# Private K-Nearest Neighbours
X_train, X_test, y_train, y_test = train_test_split(gaussianPrivateDataset.loc[:, gaussianPrivateDataset.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

training_accuracy = []
test_accuracy = []

for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of gaussian private K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of gaussian private K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
