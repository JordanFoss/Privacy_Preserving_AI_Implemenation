import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import seaborn as sns

from Data_Functions import loadDiabetesData, addLaplaceNoise, addGaussianNoise, plotFeatureImportancesDiabetes, plotConfusionMatrix

sns.set_style('darkgrid')
#Load in the data
diabetes = loadDiabetesData()

#Number of trials to run
trials = 2

diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]

# Non-private DT Neighbours
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

    tree = DecisionTreeClassifier(max_depth=3, random_state=1)
    tree.fit(X_train, y_train)
    np_training_acc.append(tree.score(X_train, y_train))
    np_testing_acc.append(tree.score(X_test, y_test))
    np_feature_importance.append(tree.feature_importances_)

print('Accuracy of non-private DT classifier on training set: {:.2f}'.format(np.mean(np_training_acc)))
print('Accuracy of non-private DT classifier on test set: {:.2f}'.format(np.mean(np_testing_acc)))
    
for index in range(len(diabetes_features)):
    np_mean_feature_importance.append(np.mean([x[index] for x in np_feature_importance]))
print("Feature importances: {}".format(np_mean_feature_importance))

plotFeatureImportancesDiabetes(tree, diabetes_features, "Non-Private Decision Tree")
plt.savefig('feature_importance')
plotConfusionMatrix(y_test, tree.predict(X_test), "Non-Private Decision Tree Confusion Matrix")

# Private Laplace Decision Tree Model
# Generate the private data set
delta = 1
epsilon = 0.1
scale = delta/epsilon

l_training_acc = []
l_testing_acc = []
l_feature_importance = []
l_mean_feature_importance = []

for i in range(trials):
    laplacePrivateDataset = addLaplaceNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale)
    training_acc = []
    testing_acc = []
    
    for train, test in kf.split(diabetes):
        X_train, X_test, y_train, y_test = (laplacePrivateDataset.loc[train, laplacePrivateDataset.columns != 'Outcome'], 
                                            laplacePrivateDataset.loc[test, laplacePrivateDataset.columns != 'Outcome'], 
                                            diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                            diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
    
        tree = DecisionTreeClassifier(max_depth=3, random_state=1)
        tree.fit(X_train, y_train)
        training_acc.append(tree.score(X_train, y_train))
        testing_acc.append(tree.score(X_test, y_test))
        l_feature_importance.append(tree.feature_importances_)

    l_training_acc.append(np.mean(training_acc))
    l_testing_acc.append(np.mean(testing_acc))
    
print('Accuracy of laplace private DT classifier on training set: {:.2f}'.format(np.mean(l_training_acc)))
print('Accuracy of laplace private DT classifier on test set: {:.2f}'.format(np.mean(l_testing_acc)))

fig, axis = plt.subplots(figsize =(10, 5))
plt.hist(l_training_acc, bins = np.linspace(np.amin(l_training_acc), np.amax(np_training_acc), 20))
plt.axvline(np.mean(np_training_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("DT Training Set Accuracy under Laplace Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(l_testing_acc, bins = np.linspace(np.amin(l_testing_acc), np.amax(np_testing_acc), 20))
plt.axvline(np.mean(np_testing_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("DT Testing Set Accuracy under Laplace Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

for index in range(len(diabetes_features)):
    l_mean_feature_importance.append(np.mean([x[index] for x in l_feature_importance]))
    
print("Feature importances: {}".format(l_mean_feature_importance))

plotFeatureImportancesDiabetes(tree, diabetes_features, "Laplace Private Decision Tree")
plt.savefig('feature_importance')
plotConfusionMatrix(y_test, tree.predict(X_test), "Laplace Private Decision Tree Confusion Matrix")

g_training_acc = []
g_testing_acc = []
g_feature_importance = []
g_mean_feature_importance = []

# Private Gaussian Logistic Regression Model
for i in range(trials):
    # Generate a copy of the data set to make private
    gaussianPrivateDataset = addGaussianNoise(diabetes.loc[:, diabetes.columns != 'Outcome'], 0, scale)
    training_acc = []
    testing_acc = []
    
    for train, test in kf.split(diabetes):
        X_train, X_test, y_train, y_test = (gaussianPrivateDataset.loc[train, gaussianPrivateDataset.columns != 'Outcome'], 
                                            gaussianPrivateDataset.loc[test, gaussianPrivateDataset.columns != 'Outcome'], 
                                            diabetes.loc[train, diabetes.columns == 'Outcome'].squeeze(), 
                                            diabetes.loc[test, diabetes.columns == 'Outcome'].squeeze())
    
        tree = DecisionTreeClassifier(max_depth=3, random_state=1)
        tree.fit(X_train, y_train)
        training_acc.append(tree.score(X_train, y_train))
        testing_acc.append(tree.score(X_test, y_test))
        g_feature_importance.append(tree.feature_importances_)

    g_training_acc.append(np.mean(training_acc))
    g_testing_acc.append(np.mean(testing_acc))
    
print('Accuracy of gaussian private DT classifier on training set: {:.2f}'.format(np.mean(g_training_acc)))
print('Accuracy of gaussian private DT classifier on test set: {:.2f}'.format(np.mean(g_testing_acc)))

fig, axis = plt.subplots(figsize =(10, 5))
plt.hist(l_training_acc, bins = np.linspace(np.amin(g_training_acc), np.amax(np_training_acc), 20))
plt.axvline(np.mean(np_training_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("DT Training Set Accuracy under Gaussian Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(l_testing_acc, bins = np.linspace(np.amin(g_testing_acc), np.amax(np_testing_acc), 20))
plt.axvline(np.mean(np_testing_acc), color='k', linestyle='dashed', linewidth=1)
plt.title("DT Testing Set Accuracy under Gaussian Noise")
plt.xlabel("Accuracy (%)")
plt.ylabel("Number of Samples")
plt.show()

for index in range(len(diabetes_features)):
    g_mean_feature_importance.append(np.mean([x[index] for x in g_feature_importance]))
    
print("Feature importances: {}".format(g_mean_feature_importance))

plotFeatureImportancesDiabetes(tree, diabetes_features, "Gaussian Private Decision Tree")
plt.savefig('feature_importance')
plotConfusionMatrix(y_test, tree.predict(X_test), "Gaussian Private Decision Tree Confusion Matrix")