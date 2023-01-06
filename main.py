import itertools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, calinski_harabasz_score, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential  # For neural network
from keras.layers import Dense  # For neural network

trainingData = pd.read_csv("CensusCanada2016Training.csv")
testData = pd.read_csv("CensusCanada2016Test.csv")

# Data cleaning
trainingData.info()
testData.info()
trainingData.columns = trainingData.columns.str.replace(" ", "")
testData.columns = testData.columns.str.replace(" ", "")
sns.histplot(data=trainingData, x="TotalHouseholds")
sns.histplot(data=trainingData, x="TotalPopulation")
plt.xlabel("Blue: Household    Orange: Population")
plt.show()
# Drop rows where total household or total population are 0
trainingData_clean = trainingData.loc[(trainingData["TotalPopulation"] != 0) & (trainingData["TotalHouseholds"] != 0)]
trainingData_clean.info()

# EDA
# Multivariate Analysis
pairPlot = sns.pairplot(data=trainingData_clean)
for ax in pairPlot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), rotation=45)  # rotate x axis labels
    ax.set_ylabel(ax.get_ylabel(), rotation=45)  # rotate y axis labels
    ax.yaxis.get_label().set_horizontalalignment('right')  # set y labels alignment
plt.subplots_adjust(left=0.13, bottom=0.13, right=0.97, top=0.95)  # adjust margin
pairPlot.fig.suptitle("Multivariate Analysis", fontsize=40)
plt.show()
# Correlation Analysis
correlationMatrix = trainingData_clean.corr()  # Returns a dataframe
plt.figure(figsize=(18, 18))
sns.heatmap(correlationMatrix, vmin=-1, vmax=1, annot=True)
plt.title("Correlation Matrix", fontsize=40)
plt.subplots_adjust(left=0.23, bottom=0.23, right=0.98, top=0.9)  # adjust margin
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right', rotation_mode="anchor")
plt.show()

# Feature Engineering
# Drop highly correlated variables: TotalPopulation and TotalHouseholdsForPeriodOfConstruction
trainingData_clean = trainingData_clean.drop(["TotalPopulation", "TotalHouseholdsForPeriodOfConstruction"],
                                             axis="columns")
testData = testData.drop(["TotalPopulation", "TotalHouseholdsForPeriodOfConstruction"], axis="columns")

# Normalization
trainingDataColumns = trainingData_clean.columns  # Back up the original column names
testDataColumns = testData.columns  # Back up the original column names
standardScaler = StandardScaler()
trainingData_clean_nm = standardScaler.fit_transform(trainingData_clean)
trainingData_clean_nm = pd.DataFrame(trainingData_clean_nm, columns=trainingDataColumns)
testData_nm = standardScaler.fit_transform(testData)
testData_nm = pd.DataFrame(testData_nm, columns=testDataColumns)

# K-mean clustering
# Find the best K to use
# Elbow method
numOfClusters = list(range(2, 16))
inertia = []
for num in numOfClusters:
    kMeanClusterModel = KMeans(n_clusters=num)
    kMeanClusterModel.fit_predict(trainingData_clean_nm.drop("MedianHouseholdIncome(CurrentYear$)", axis="columns"))
    inertia.append(kMeanClusterModel.inertia_)
sns.lineplot(x=numOfClusters, y=inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
# Silhouette method
silhouette = []
for num in numOfClusters:
    kMeanClusterModel = KMeans(n_clusters=num)
    kMeanClusterModel.fit_predict(trainingData_clean_nm.drop("MedianHouseholdIncome(CurrentYear$)", axis="columns"))
    silhouetteScore = silhouette_score(
        trainingData_clean_nm.drop("MedianHouseholdIncome(CurrentYear$)", axis="columns"),
        kMeanClusterModel.labels_, metric='euclidean')
    silhouette.append(silhouetteScore)
sns.lineplot(x=numOfClusters, y=silhouette, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
# pseudo-F Statistic method
pseudoF = []
for num in numOfClusters:
    kMeanClusterModel = KMeans(n_clusters=num)
    kMeanClusterModel.fit_predict(trainingData_clean_nm.drop("MedianHouseholdIncome(CurrentYear$)", axis="columns"))
    pseudoFScore = calinski_harabasz_score(
        trainingData_clean_nm.drop("MedianHouseholdIncome(CurrentYear$)", axis="columns"),
        kMeanClusterModel.labels_)
    pseudoF.append(pseudoFScore)
sns.lineplot(x=numOfClusters, y=pseudoF, marker='o')
plt.title("Pseudo-F")
plt.xlabel("Number of Clusters")
plt.ylabel("Pseudo F score")
plt.show()
# As we can see, all 3 methods suggest that 3 is the best cluster size

# Fit the model using k=3
kMeanClusterModel = KMeans(n_clusters=3)
kMeanClusterModel.fit_predict(trainingData_clean_nm.drop("MedianHouseholdIncome(CurrentYear$)", axis="columns"))
trainingData_clean["Cluster"] = kMeanClusterModel.labels_  # Save cluster to un-normalized dataset
# Predict clusters for test data
prediction = kMeanClusterModel.predict(testData_nm)
testData["Cluster"] = prediction  # Save cluster to un-normalized dataset
# Check cluster distribution of training data and test data
sns.countplot(x="Cluster", data=trainingData_clean)
plt.title("Distribution of clusters in training dataset")
plt.show()
sns.countplot(x="Cluster", data=testData)
plt.title("Distribution of clusters in test dataset")
plt.show()
# As we can see, the cluster distributions in training and test dataset are very similar, which means our clustering is
# good and consistent

# EDA of training dataset by cluster
# Distribution of target variable "MedianHouseholdIncome(CurrentYear$)"
# Histogram
sns.histplot(data=trainingData_clean, x="MedianHouseholdIncome(CurrentYear$)", hue="Cluster",
             palette=['green', 'yellow', 'red'])
plt.xlim(0, 300000)
plt.title("Distribution of income by cluster")
plt.show()
# Boxplot
sns.boxplot(data=trainingData_clean, x="Cluster", y="MedianHouseholdIncome(CurrentYear$)")
plt.ylim(0, 300000)
plt.subplots_adjust(left=0.16, bottom=0.12, right=0.95, top=0.9)  # adjust margin
plt.title("Distribution of income by cluster")
plt.show()


# It is very important to stop here and take a look at the cluster distribution graph. We change the number below to
# assign the records of the most populated cluster into cluster0, second most populated cluster records into cluster1
# and the least populated cluster records into cluster 2. Eg, if most records are assigned cluster number 2, then in
# the code below we assign all records with a cluster number 2 in to cluster0_training and cluster0_test. By doing this,
# we are making sure that cluster 0 is always the most populated cluster, cluster 1 is always the second most populated
# cluster and cluster 2 the least populated cluster.


# Divide training data by clusters
cluster0_training = trainingData_clean.loc[trainingData_clean["Cluster"] == 0].drop("Cluster", axis="columns")
cluster1_training = trainingData_clean.loc[trainingData_clean["Cluster"] == 2].drop("Cluster", axis="columns")
cluster2_training = trainingData_clean.loc[trainingData_clean["Cluster"] == 1].drop("Cluster", axis="columns")
# Divide test data by cluster
cluster0_test = testData.loc[testData["Cluster"] == 0].drop("Cluster", axis="columns")
cluster1_test = testData.loc[testData["Cluster"] == 2].drop("Cluster", axis="columns")
cluster2_test = testData.loc[testData["Cluster"] == 1].drop("Cluster", axis="columns")

# Cluster 0 prediction
X_cluster0_training = cluster0_training.drop("MedianHouseholdIncome(CurrentYear$)", axis="columns")
y_cluster0_training = cluster0_training["MedianHouseholdIncome(CurrentYear$)"]
X_train, X_test, y_train, y_test = train_test_split(X_cluster0_training, y_cluster0_training, test_size=0.2,
                                                    random_state=0)

# First we try KNN, since it does not assume functional form
# Since KNN is a distance bases algorithm, we need to normalize our data again
X_train_nm = standardScaler.fit_transform(X_train)
X_train_nm = pd.DataFrame(X_train_nm, columns=X_train.columns)
y_train_nm = standardScaler.fit_transform(pd.DataFrame(y_train))
X_test_nm = standardScaler.fit_transform(X_test)
X_test_nm = pd.DataFrame(X_test_nm, columns=X_test.columns)
y_test_nm = standardScaler.fit_transform(pd.DataFrame(y_test))
# Use cross validation to find best k
scoring = "neg_mean_squared_error"  # The closer to 0 the better
listOfScores = []  # This is basically a in-house grid search algorithm
k = 1
while k <= 100:
    knn_cv = KNeighborsRegressor(n_neighbors=k, weights="distance")  # Distance weighted
    cv_scores = cross_val_score(knn_cv, X_train_nm, y_train, cv=10, scoring=scoring)  # Perform 10-fold cross validation
    cv_scores = np.mean(cv_scores)
    listOfScores.append([cv_scores, "k = " + str(k)])
    k = k + 1
print(max(listOfScores))  # Max evaluates the first number in each list in a list, best k is 14
# Use k = 14 to train our model
knnModel = KNeighborsRegressor(n_neighbors=14, weights="distance")  # Distance weighted
knnModel.fit(X_train_nm, y_train)
# Evaluate the model using test data
prediction = knnModel.predict(X_test_nm)
r2 = r2_score(y_test, prediction)
print("KNN test r2 is: " + str(r2))
# r2 is bad, only 0.3903, we need to try something else

# Linear regression
lRModel = LinearRegression()
lRModel.fit(X_train, y_train)
prediction = lRModel.predict(X_test)
r2 = r2_score(y_test, prediction)
print("Linear regression test r2 is: " + str(r2))
# r2 is even worse. Only 0.3869

# CART
# Grid search cross validation to find the best hyperparameters
param_grid = {
    "max_depth": list(range(2, 51)),
    "min_samples_split": list(range(2, 101))
}
gridSearch = GridSearchCV(estimator=DecisionTreeRegressor(random_state=1), param_grid=param_grid, cv=5, n_jobs=-1)
gridSearch.fit(X_train, y_train)
gridSearch.best_score_
gridSearch.best_params_
# Best max depth is 2 and best minimum samples split is 2
bestTreeRegressor = gridSearch.best_estimator_
prediction = bestTreeRegressor.predict(X_test)
r2 = r2_score(y_test, prediction)
print("CART test r2 is: " + str(r2))
# r2 is getting worse and worse. This is humiliating. 0.3165

# Neural Network
# We need to use normalized data generated for KNN previously
# Cross validation to find hyperparameters
from keras.wrappers.scikit_learn import KerasRegressor


def create_model(neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=12, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(1, kernel_initializer="uniform", activation="relu"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=16, verbose=2)
neuronList = [1, 5, 10, 15, 20, 30, 40, 50, 100]
param_grid = {
    "neurons": neuronList
}
gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
gridSearch.fit(X_train_nm, y_train_nm)
gridSearch.best_score_
gridSearch.best_params_
# Best neuron number is 20
model = Sequential()
model.add(Dense(20, input_dim=12, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="relu"))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train_nm, y_train_nm, epochs=100, batch_size=16, verbose=2)
prediction = model.predict(X_test_nm)
prediction
r2 = r2_score(y_test_nm, prediction)
print("Neural Network test r2 is: " + str(r2))
# As expected, the r2 become even lower. 0.2612

# Dimension reduction using linear regression
iteration = 1
subsetList = []
r2List = []
lRModel = LinearRegression()
featureList = X_train.columns
for length in range(1, len(featureList) + 1):
    for subset in itertools.combinations(featureList, length):
        print("Training iteration " + str(iteration))
        aList = list(subset)
        subsetList.append(aList)
        lRModel.fit(X_train[aList], y_train)
        prediction = lRModel.predict(X_test[aList])
        r2 = r2_score(y_test, prediction)
        r2List.append(r2)
        iteration = iteration + 1
subsetAndR2 = pd.DataFrame({"SubsetList": subsetList, "r2": r2List})
bestSubset = subsetAndR2.loc[subsetAndR2['r2'].idxmax(), 'SubsetList']
bestR2 = subsetAndR2['r2'].max()
print("The best subset using linear regression is: \n" + str(bestSubset) + ".\nThe associated r2 is: " + str(bestR2))
# Best features are:
# 'TotalHouseholds'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBefore1961'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1961And1980'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1991And2000'
# 'TotalHouseholdsForStructureTypeOtherDwellingTypes'
# 'DwellingsbyTenureRenter'
# Best r2 using KNN and the best subset is 0.4028261096557223

# Dimension reduction using KNN
iteration = 1
subsetList = []
r2List = []
featureList = X_train.columns
knnModel = KNeighborsRegressor(n_neighbors=14, weights="distance")
for length in range(1, len(featureList) + 1):
    for subset in itertools.combinations(featureList, length):
        print("Training iteration " + str(iteration))
        aList = list(subset)
        subsetList.append(aList)
        knnModel.fit(X_train_nm[aList], y_train)
        # Evaluate the model using test data
        prediction = knnModel.predict(X_test_nm[aList])
        r2 = r2_score(y_test, prediction)
        r2List.append(r2)
        iteration = iteration + 1
subsetAndR2 = pd.DataFrame({"SubsetList": subsetList, "r2": r2List})
bestSubset = subsetAndR2.loc[subsetAndR2['r2'].idxmax(), 'SubsetList']
bestR2 = subsetAndR2['r2'].max()
print("The best subset using KNN is: \n" + str(bestSubset) + ".\nThe associated r2 is: " + str(bestR2))
# Best features are:
# 'TotalHouseholds'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBefore1961'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1961And1980'
# 'TotalHouseholdsForStructureTypeOtherDwellingTypes'
# 'DwellingsbyTenureRenter'
# Best r2 using KNN and the best subset is 0.4852

# Conclusion for cluster 0 (The most populated cluster):
# Use the following features
# 'TotalHouseholds'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBefore1961'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1961And1980'
# 'TotalHouseholdsForStructureTypeOtherDwellingTypes'
# 'DwellingsbyTenureRenter'
# Model: KNN with k = 14
# Best possible r2 using KNN and the best subset is 0.4852

# Now train the model and predict the final result
# First modify the training and test features
X_cluster0_training = cluster0_training[bestSubset]
cluster0_test = cluster0_test[bestSubset]
y_cluster0_training = cluster0_training["MedianHouseholdIncome(CurrentYear$)"]
# Next we normalize the dataset to prepare for KNN
X_train_nm = standardScaler.fit_transform(X_cluster0_training)
X_train_nm = pd.DataFrame(X_train_nm, columns=X_cluster0_training.columns)
X_test_nm = standardScaler.fit_transform(cluster0_test)
X_test_nm = pd.DataFrame(X_test_nm, columns=cluster0_test.columns)
# Model training
knnModel = KNeighborsRegressor(n_neighbors=14, weights="distance")
knnModel.fit(X_train_nm, y_cluster0_training)
cluster0_prediction = knnModel.predict(X_test_nm)
cluster0_test["prediction"] = cluster0_prediction

# Cluster 1 prediction

X_cluster1_training = cluster1_training.drop("MedianHouseholdIncome(CurrentYear$)", axis="columns")
y_cluster1_training = cluster1_training["MedianHouseholdIncome(CurrentYear$)"]
X_train, X_test, y_train, y_test = train_test_split(X_cluster1_training, y_cluster1_training, test_size=0.2,
                                                    random_state=0)

# First we try KNN, since it does not assume functional form
# Since KNN is a distance bases algorithm, we need to normalize our data again
X_train_nm = standardScaler.fit_transform(X_train)
X_train_nm = pd.DataFrame(X_train_nm, columns=X_train.columns)
X_test_nm = standardScaler.fit_transform(X_test)
X_test_nm = pd.DataFrame(X_test_nm, columns=X_test.columns)
# Use cross validation to find best k
scoring = "neg_mean_squared_error"  # The closer to 0 the better
listOfScores = []  # This is basically a in-house grid search algorithm
k = 1
while k <= 100:
    knn_cv = KNeighborsRegressor(n_neighbors=k, weights="distance")  # Distance weighted
    cv_scores = cross_val_score(knn_cv, X_train_nm, y_train, cv=10, scoring=scoring)  # Perform 10-fold cross validation
    cv_scores = np.mean(cv_scores)
    listOfScores.append([cv_scores, "k = " + str(k)])
    k = k + 1
print(max(listOfScores))  # Max evaluates the first number in each list in a list, best k is 20
# Use k = 20 to train our model
knnModel = KNeighborsRegressor(n_neighbors=20, weights="distance")  # Distance weighted
knnModel.fit(X_train_nm, y_train)
# Evaluate the model using test data
prediction = knnModel.predict(X_test_nm)
r2 = r2_score(y_test, prediction)
print("KNN test r2 is: " + str(r2))
# r2 is 0.3091, we need to try something else

# Linear regression
lRModel = LinearRegression()
lRModel.fit(X_train, y_train)
prediction = lRModel.predict(X_test)
r2 = r2_score(y_test, prediction)
print("Linear regression test r2 is: " + str(r2))
# r2 is better. Only 0.3279

# CART
# Grid search cross validation to find the best hyperparameters
param_grid = {
    "max_depth": list(range(2, 51)),
    "min_samples_split": list(range(2, 101))
}
gridSearch = GridSearchCV(estimator=DecisionTreeRegressor(random_state=1), param_grid=param_grid, cv=5, n_jobs=-1)
gridSearch.fit(X_train, y_train)
gridSearch.best_score_
gridSearch.best_params_
# Best max depth is 2 and best minimum samples split is 2
bestTreeRegressor = gridSearch.best_estimator_
prediction = bestTreeRegressor.predict(X_test)
r2 = r2_score(y_test, prediction)
print("CART test r2 is: " + str(r2))
# r2 is very bad. 0.1628

# Neural Network
# We need to use normalized data generated for KNN previously
# Cross validation to find hyperparameters
from keras.wrappers.scikit_learn import KerasRegressor


def create_model(neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=12, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(1, kernel_initializer="uniform", activation="relu"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=16, verbose=2)
neuronList = [1, 5, 10, 15, 20, 30, 40, 50, 100]
param_grid = {
    "neurons": neuronList
}
gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
gridSearch.fit(X_train_nm, y_train_nm)
gridSearch.best_score_
gridSearch.best_params_
# Best neuron number is 20
model = Sequential()
model.add(Dense(20, input_dim=12, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="relu"))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train_nm, y_train_nm, epochs=100, batch_size=16, verbose=2)
prediction = model.predict(X_test_nm)
prediction
r2 = r2_score(y_test_nm, prediction)
print("Neural Network test r2 is: " + str(r2))
# As expected, the r2 become even lower. 0.2114

# Dimension reduction using linear regression
iteration = 1
subsetList = []
r2List = []
lRModel = LinearRegression()
featureList = X_train.columns
for length in range(1, len(featureList) + 1):
    for subset in itertools.combinations(featureList, length):
        print("Training iteration " + str(iteration))
        aList = list(subset)
        subsetList.append(aList)
        lRModel.fit(X_train[aList], y_train)
        prediction = lRModel.predict(X_test[aList])
        r2 = r2_score(y_test, prediction)
        r2List.append(r2)
        iteration = iteration + 1
subsetAndR2 = pd.DataFrame({"SubsetList": subsetList, "r2": r2List})
bestSubset = subsetAndR2.loc[subsetAndR2['r2'].idxmax(), 'SubsetList']
bestR2 = subsetAndR2['r2'].max()
print("The best subset using linear regression is: \n" + str(bestSubset) + ".\nThe associated r2 is: " + str(bestR2))
# Best features are:
# 'TotalHouseholds',
# 'TotalHouseholdsForPeriodOfConstructionBuiltBefore1961',
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1961And1980',
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1981And190',
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1991And2000',
# 'TotalHouseholdsForStructureTypeHouses',
# 'TotalHouseholdsforTenure',
# 'DwellingsbyTenureOwner',
# 'DwellingsbyTenureRenter'
# Best r2 using linear regression and the best subset is 0.3373

# Dimension reduction using KNN
iteration = 1
subsetList = []
r2List = []
featureList = X_train.columns
knnModel = KNeighborsRegressor(n_neighbors=20, weights="distance")
for length in range(1, len(featureList) + 1):
    for subset in itertools.combinations(featureList, length):
        print("Training iteration " + str(iteration))
        aList = list(subset)
        subsetList.append(aList)
        knnModel.fit(X_train_nm[aList], y_train)
        # Evaluate the model using test data
        prediction = knnModel.predict(X_test_nm[aList])
        r2 = r2_score(y_test, prediction)
        r2List.append(r2)
        iteration = iteration + 1
subsetAndR2 = pd.DataFrame({"SubsetList": subsetList, "r2": r2List})
bestSubset = subsetAndR2.loc[subsetAndR2['r2'].idxmax(), 'SubsetList']
bestR2 = subsetAndR2['r2'].max()
print("The best subset using KNN is: \n" + str(bestSubset) + ".\nThe associated r2 is: " + str(bestR2))
# Best features are:
# 'TotalHouseholdsForPeriodOfConstructionBuiltBefore1961',
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1961And1980',
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1981And190',
# 'TotalHouseholdsForStructureTypeHouses',
# 'TotalHouseholdsForStructureTypeApartment,BuildingLowAndHighRise',
# 'DwellingsbyTenureRenter'
# Best r2 using KNN and the best subset is 0.3570

# Conclusion for cluster 1 (The second most populated cluster):
# Use the following features
# 'TotalHouseholdsForPeriodOfConstructionBuiltBefore1961',
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1961And1980',
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1981And190',
# 'TotalHouseholdsForStructureTypeHouses',
# 'TotalHouseholdsForStructureTypeApartment,BuildingLowAndHighRise',
# 'DwellingsbyTenureRenter'
# Model: KNN with k = 20
# Best possible r2 using KNN and the best subset is 0.3570

# Now train the model and predict the final result
# First modify the training and test features
X_cluster1_training = cluster1_training[bestSubset]
cluster1_test = cluster1_test[bestSubset]
y_cluster1_training = cluster1_training["MedianHouseholdIncome(CurrentYear$)"]
# Next we normalize the dataset to prepare for KNN
X_train_nm = standardScaler.fit_transform(X_cluster1_training)
X_train_nm = pd.DataFrame(X_train_nm, columns=X_cluster1_training.columns)
X_test_nm = standardScaler.fit_transform(cluster1_test)
X_test_nm = pd.DataFrame(X_test_nm, columns=cluster1_test.columns)
# Model training
knnModel = KNeighborsRegressor(n_neighbors=20, weights="distance")
knnModel.fit(X_train_nm, y_cluster1_training)
cluster1_prediction = knnModel.predict(X_test_nm)
cluster1_test["prediction"] = cluster1_prediction

# Cluster 2 prediction
X_cluster2_training = cluster2_training.drop("MedianHouseholdIncome(CurrentYear$)", axis="columns")
y_cluster2_training = cluster2_training["MedianHouseholdIncome(CurrentYear$)"]
X_train, X_test, y_train, y_test = train_test_split(X_cluster2_training, y_cluster2_training, test_size=0.2,
                                                    random_state=0)

# First we try KNN, since it does not assume functional form
# Since KNN is a distance bases algorithm, we need to normalize our data again
X_train_nm = standardScaler.fit_transform(X_train)
X_train_nm = pd.DataFrame(X_train_nm, columns=X_train.columns)
X_test_nm = standardScaler.fit_transform(X_test)
X_test_nm = pd.DataFrame(X_test_nm, columns=X_test.columns)
# Use cross validation to find best k
scoring = "neg_mean_squared_error"  # The closer to 0 the better
listOfScores = []  # This is basically a in-house grid search algorithm
k = 1
while k <= 100:
    knn_cv = KNeighborsRegressor(n_neighbors=k, weights="distance")  # Distance weighted
    cv_scores = cross_val_score(knn_cv, X_train_nm, y_train, cv=10, scoring=scoring)  # Perform 10-fold cross validation
    cv_scores = np.mean(cv_scores)
    listOfScores.append([cv_scores, "k = " + str(k)])
    k = k + 1
print(max(listOfScores))  # Max evaluates the first number in each list in a list, best k is 16
# Use k = 16 to train our model
knnModel = KNeighborsRegressor(n_neighbors=16, weights="distance")  # Distance weighted
knnModel.fit(X_train_nm, y_train)
# Evaluate the model using test data
prediction = knnModel.predict(X_test_nm)
r2 = r2_score(y_test, prediction)
print("KNN test r2 is: " + str(r2))
# r2 is 0.3545, we need to try something else

# Linear regression
lRModel = LinearRegression()
lRModel.fit(X_train, y_train)
prediction = lRModel.predict(X_test)
r2 = r2_score(y_test, prediction)
print("Linear regression test r2 is: " + str(r2))
# r2 is better. 0.3881

# CART
# Grid search cross validation to find the best hyperparameters
param_grid = {
    "max_depth": list(range(2, 51)),
    "min_samples_split": list(range(2, 101))
}
gridSearch = GridSearchCV(estimator=DecisionTreeRegressor(random_state=1), param_grid=param_grid, cv=5, n_jobs=-1)
gridSearch.fit(X_train, y_train)
gridSearch.best_score_
gridSearch.best_params_
# Best max depth is 2 and best minimum samples split is 2
bestTreeRegressor = gridSearch.best_estimator_
prediction = bestTreeRegressor.predict(X_test)
r2 = r2_score(y_test, prediction)
print("CART test r2 is: " + str(r2))
# r2 is getting worse. 0.3494

# Neural Network
# We need to use normalized data generated for KNN previously
# Cross validation to find hyperparameters
from keras.wrappers.scikit_learn import KerasRegressor


def create_model(neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=12, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(1, kernel_initializer="uniform", activation="relu"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=16, verbose=2)
neuronList = [1, 5, 10, 15, 20, 30, 40, 50, 100]
param_grid = {
    "neurons": neuronList
}
gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
gridSearch.fit(X_train_nm, y_train_nm)
gridSearch.best_score_
gridSearch.best_params_
# Best neuron number is 20
model = Sequential()
model.add(Dense(20, input_dim=12, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="relu"))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train_nm, y_train_nm, epochs=100, batch_size=16, verbose=2)
prediction = model.predict(X_test_nm)
prediction
r2 = r2_score(y_test_nm, prediction)
print("Neural Network test r2 is: " + str(r2))
# As expected, the r2 become even lower. 0.1928

# Dimension reduction using linear regression
iteration = 1
subsetList = []
r2List = []
lRModel = LinearRegression()
featureList = X_train.columns
for length in range(1, len(featureList) + 1):
    for subset in itertools.combinations(featureList, length):
        print("Training iteration " + str(iteration))
        aList = list(subset)
        subsetList.append(aList)
        lRModel.fit(X_train[aList], y_train)
        prediction = lRModel.predict(X_test[aList])
        r2 = r2_score(y_test, prediction)
        r2List.append(r2)
        iteration = iteration + 1
subsetAndR2 = pd.DataFrame({"SubsetList": subsetList, "r2": r2List})
bestSubset = subsetAndR2.loc[subsetAndR2['r2'].idxmax(), 'SubsetList']
bestR2 = subsetAndR2['r2'].max()
print("The best subset using linear regression is: \n" + str(bestSubset) + ".\nThe associated r2 is: " + str(bestR2))
# Best features are:
# 'TotalHouseholdsForPeriodOfConstructionBuiltBefore1961'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1961And1980'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1981And190'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween2001And2005'
# 'TotalHouseholdsForStructureTypeHouses'
# 'TotalHouseholdsForStructureTypeApartment,BuildingLowAndHighRise'
# 'TotalHouseholdsforTenure'
# 'DwellingsbyTenureRenter'
# Best r2 using linear regression and the best subset is 0.4108

# Dimension reduction using KNN
iteration = 1
subsetList = []
r2List = []
featureList = X_train.columns
knnModel = KNeighborsRegressor(n_neighbors=16, weights="distance")
for length in range(1, len(featureList) + 1):
    for subset in itertools.combinations(featureList, length):
        print("Training iteration " + str(iteration))
        aList = list(subset)
        subsetList.append(aList)
        knnModel.fit(X_train_nm[aList], y_train)
        # Evaluate the model using test data
        prediction = knnModel.predict(X_test_nm[aList])
        r2 = r2_score(y_test, prediction)
        r2List.append(r2)
        iteration = iteration + 1
subsetAndR2 = pd.DataFrame({"SubsetList": subsetList, "r2": r2List})
bestSubset = subsetAndR2.loc[subsetAndR2['r2'].idxmax(), 'SubsetList']
bestR2 = subsetAndR2['r2'].max()
print("The best subset using KNN is: \n" + str(bestSubset) + ".\nThe associated r2 is: " + str(bestR2))
# Best features are:
# 'TotalHouseholds'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1961And1980'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1981And190'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween2001And2005'
# 'TotalHouseholdsForStructureTypeHouses'
# 'TotalHouseholdsForStructureTypeApartment,BuildingLowAndHighRise'
# 'DwellingsbyTenureOwner',
# 'DwellingsbyTenureRenter'
# Best r2 using KNN and the best subset is 0.4852

# Conclusion for cluster 2 (The least populated cluster):
# Use the following features
# 'TotalHouseholds'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1961And1980'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween1981And190'
# 'TotalHouseholdsForPeriodOfConstructionBuiltBetween2001And2005'
# 'TotalHouseholdsForStructureTypeHouses'
# 'TotalHouseholdsForStructureTypeApartment,BuildingLowAndHighRise'
# 'DwellingsbyTenureOwner',
# 'DwellingsbyTenureRenter'
# Model: KNN with k = 16
# Best possible r2 using KNN and the best subset is 0.4852

# Now train the model and predict the final result
# First modify the training and test features
X_cluster2_training = cluster2_training[bestSubset]
cluster2_test = cluster2_test[bestSubset]
y_cluster2_training = cluster2_training["MedianHouseholdIncome(CurrentYear$)"]
# Next we normalize the dataset to prepare for KNN
X_train_nm = standardScaler.fit_transform(X_cluster2_training)
X_train_nm = pd.DataFrame(X_train_nm, columns=X_cluster2_training.columns)
X_test_nm = standardScaler.fit_transform(cluster2_test)
X_test_nm = pd.DataFrame(X_test_nm, columns=cluster2_test.columns)
# Model training
knnModel = KNeighborsRegressor(n_neighbors=16, weights="distance")
knnModel.fit(X_train_nm, y_cluster2_training)
cluster2_prediction = knnModel.predict(X_test_nm)
cluster2_test["prediction"] = cluster2_prediction

finalPrediction = pd.concat([cluster0_test, cluster1_test, cluster2_test])
finalPrediction.sort_index(inplace=True)
finalPrediction = finalPrediction["prediction"]

# File creation
if not os.path.exists("Team15predictions.txt"):  # Check if file already exist
    f1 = open("Team15predictions.txt", "x")
    for value in finalPrediction:
        f1.write(str(value) + "\n")
    f1.close()
