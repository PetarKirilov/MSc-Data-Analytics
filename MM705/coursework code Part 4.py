# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:52:52 2017

@author: Rocky
"""
import math
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from matplotlib import style
style.use("ggplot")
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from sklearn.decomposition import PCA


file = "BrexitVotingAndWardDemographicData.csv"
os.chdir("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Coursework/FINAL")
brexit = pd.read_csv(file)

brex = brexit[["CountingArea", "Outcome", "Males", "Females"]]

def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

data, _ = number_encode_features(brexit)
mask = np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(data.corr(), mask = mask, square=True, cmap="RdYlGn")
ax.set_title("Heatmap")
plt.savefig("heatmap.png")
plt.show()

pcaa = PCA(n_components = 69, random_state = 15)
encoded_data, encoders = number_encode_features(brexit)
pcaa.fit(encoded_data)
print(pcaa.explained_variance_ratio_)
get_params(pcaa)

encoded_data, encoders = number_encode_features(brexit)
brexit_target = encoded_data["Outcome"].values
brexit_data = encoded_data.drop("Outcome", axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(brexit_data, brexit_target, test_size = 0.2, random_state = 15)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["Outcome"].classes_, yticklabels=encoders["Outcome"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("Logistic Regression Confusion Matrix")
plt.show()
metrics.cohen_kappa_score(y_test, y_pred)



logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(logreg.score(X_test, y_test))
cv_scores = cross_val_score(logreg, brexit_data, brexit_target, cv = 5)
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


#FIX  draw up the histogram
fig = plt.figure(figsize=(18,13))
cols = 3
rows = (float(brexit.shape[1]) / cols)
for i, column in enumerate(brexit.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if brexit.dtypes[column] == np.object:
        brexit[column].value_counts().plot(kind="bar", axes=ax, color = "lightgreen")
    else:
        brexit[column].hist(axes=ax, color = "lightgreen")
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.3, wspace=0.2)



########################################################################################################################################
#TASK2
file = "users.csv"
os.chdir("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Coursework/FINAL")
emi = pd.read_csv(file)
emi.isnull().sum()/len(emi)*100
#clean missing values
emi = pd.DataFrame.dropna(emi)
emiq = emi.iloc[:,8:27]



mask = np.zeros_like(emiq.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(emiq.corr(), mask = mask, square=True, cmap="RdYlGn")
ax.set_title("EMI Questions Heatmap")
plt.savefig("EMI Heatmap.png")
plt.show()

bins = [1,20,60,99]
group_names = ["Teenager", "Adult", "Elderly"]
categories = pd.cut(emi["AGE"], bins, labels=group_names)
emi['categories'] = pd.cut(emi['AGE'], bins, labels=group_names)
pd.value_counts(emi['categories'])



mask = np.zeros_like(emi[emi["categories"] == "Teenager"].corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(emi[emi["categories"] == "Teenager"].corr(), mask = mask, square=True, cmap="RdYlGn")
ax.set_title("EMI Teen Questions Heatmap")
plt.savefig("EMI Teen Heatmap.png")
plt.show()


mask = np.zeros_like(emi[emi["categories"] == "Adult"].corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(emi[emi["categories"] == "Adult"].corr(), mask = mask, square=True, cmap="RdYlGn")
ax.set_title("EMI Adult Questions Heatmap")
plt.savefig("EMI Adult Heatmap.png")
plt.show()

mask = np.zeros_like(emi[emi["categories"] == "Elderly"].corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(emi[emi["categories"] == "Elderly"].corr(), mask = mask, square=True, cmap="RdYlGn")
ax.set_title("EMI Elderly Questions Heatmap")
plt.savefig("EMI Elderly Heatmap.png")
plt.show()

ks = range(1, 19)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(emiq)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#seems 3 is good from the elbow graph
clust = KMeans(n_clusters=3)
clust.fit(emiq)

centroids = clust.cluster_centers_
labels = clust.labels_

print ( "centroids : ")
print (centroids)
print ("labels : ")
print (labels)

colors = ["g.","r.","c.","y."]

color = ["g", "r", "b"]

c = Counter(labels)


fig = figure()
ax = fig.gca(projection='3d')


for i in range(len(emiq)):
    print("coordinate:",emiq[i], "label:", labels[i])
    print ("i : ",i)
    print ("color[labels[i]] : ",color[labels[i]])
    ax.scatter(emiq[i][0], emiq[i][1], emiq[i][2], c=color[labels[i]])


for cluster_number in range(cluster_num):
  print("Cluster {} contains {} samples".format(cluster_number, c[cluster_number]))

ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2], marker = "x", s=150, linewidths = 5, zorder = 100, c=color)

plt.show()




###################################################################################################################
#TASK 4
os.chdir("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM705 - Multivariate Analysis and Statistical Modelling OPTIONAL SEM 2 20CR/Coursework/FINAL")
column_name = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
adult1 = pd.read_csv("adult.test.csv", names = column_name, index_col=False, sep=',\s', na_values=["?"], skiprows=1)
adult2 = pd.read_csv("adult.data.csv", names = column_name, index_col=False, sep=',\s', na_values=["?"])
adult1.income.unique()
#remove the dots
adult1.income = adult1.income.replace([">50K."], ">50K")
adult1.income = adult1.income.replace(["<=50K."], "<=50K")
#concatinate the two dfs
frame = [adult1, adult2]
adult = pd.concat(frame)
adult.income.unique()
#missing values proportions
adult.isnull().sum()/len(adult)*100
#clean missing values
adult = pd.DataFrame.dropna(adult)
adult.shape
#draw pairplot - takes a while and is quite big
#sns.pairplot(adult, hue = "gross_income")

#draw up the histogram
fig = plt.figure(figsize=(20,15))
cols = 5
rows = (float(adult.shape[1]) / cols)
for i, column in enumerate(adult.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if adult.dtypes[column] == np.object:
        adult[column].value_counts().plot(kind="bar", axes=ax, color = "lightgreen")
    else:
        adult[column].hist(axes=ax, color = "lightgreen")
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.savefig("histdist.png")

#correlation
# Encode the categorical features as numbers
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

# Calculate the correlation and plot it
data, _ = number_encode_features(adult)
mask = np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(data.corr(), mask = mask, square=True, cmap="RdYlGn")
ax.set_title("Adult Heatmap")
plt.show()
#look at education and education-num
adult[["education", "education-num"]].head
#delete education as it represents the same as education-num
del adult["education"]
adult[["sex", "relationship"]].head
 
adult.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['not married','married','married','married',
              'not married','not married','not married'], inplace = True)


adult["income"].value_counts()[0] / adult.shape[0]
#75% for under 50K
adult["income"].value_counts()[1] / adult.shape[0]


#for col in category_col:
#    b, c = np.unique(adult[col], return_inverse=True) 
#    adult[col] = c


#identify best parameters
#col_names = adult.columns
#param=[]
#correlation=[]
#abs_corr=[]
#for c in col_names:
#    #Check if binary or continuous
#    if c != "income":
#        if len(adult[c].unique()) <= 2:
#            corr = spearmanr(adult['income'],adult[c])[0]
#        else:
##            corr = pointbiserialr(adult['income'],adult[c])[0]
 #       param.append(c)
 #       correlation.append(corr)
 #       abs_corr.append(abs(corr))
#Create dataframe for visualization
#param_df=pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr})
##Sort by absolute correlation
#param_df=param_df.sort_values(by=['abs_corr'], ascending=False)
##Set parameter name as index
#param_df=param_df.set_index('parameter')
#param_df
##get best features
#best_features=param_df.index[0:4].values
#print('Best features:\t',best_features)#



#encode the data and split into test and train set
encoded_data, encoders = number_encode_features(adult)
adult_target = encoded_data["income"].values
adult_data = encoded_data.drop("income", axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(adult_data, adult_target, test_size = 0.2, random_state = 15, stratify=adult_target)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#KNN 
#Find the optimal number of neighbors
neighbors = np.arange(1, 20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
#hyperparameter tuning - it takes around 20 minutes
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1,20)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv = 5)
knn_cv.fit(X_train, y_train)
knn_cv.best_params_
knn_cv.best_score_

#specify the kNN parameters
knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))
cv_scores = cross_val_score(knn, adult_data, adult_target, cv = 5)
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["income"].classes_, yticklabels=encoders["income"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("kNN Confusion Matrix")
plt.show()
print ("F1 score: %f" % metrics.f1_score(y_test, y_pred))
metrics.cohen_kappa_score(y_test, y_pred)

y_pred_prob = knn.predict_proba(X_test)[:,1]
# Gen ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, color = "green")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.show()
y_pred_prob = knn.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(knn, scaler.transform(adult_data), adult_target, cv = 5,scoring = "roc_auc")
# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(np.mean(cv_auc)))


#LOGISTIC REG
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["income"].classes_, yticklabels=encoders["income"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("Logistic Regression Confusion Matrix")
print ("F1 score: %f" % metrics.f1_score(y_test, y_pred))
plt.show()
metrics.cohen_kappa_score(y_test, y_pred)


plt.figure(figsize=(12,12))
coefs = pd.Series(logreg.coef_[0], index=encoded_data.drop("income", axis = 1).columns)
coefs.sort()
plt.subplot(2,1,2)
coefs.plot(kind="bar", color="lightgreen")
plt.title('Contribution to Model')
plt.show()

from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, color = "green")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg,scaler.transform(adult_data), adult_target, cv = 5,scoring = "roc_auc")
# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(np.mean(cv_auc)))



binary_data = pd.get_dummies(adult)
# Let's fix the Target as it will be converted to dummy vars too
binary_data["income"] = binary_data["income_>50K"]
del binary_data["income_<=50K"]
del binary_data["income_>50K"]
binary_target = binary_data["income"].values
binary_d = binary_data.drop("income", axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(binary_d, binary_target, train_size=0.80, random_state = 15, stratify = binary_target)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred),cm)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["income"].classes_, yticklabels=encoders["income"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("Confusion Matrix of Binary Features")
plt.show()
print ("F1 score: %f" % metrics.f1_score(y_test, y_pred))
metrics.cohen_kappa_score(y_test, y_pred)
plt.figure(figsize=(12,12))
coefs = pd.Series(logreg.coef_[0], index=binary_data.drop("income", axis = 1).columns)
coefs.sort()
ax = plt.subplot(2,1,2)
coefs.plot(kind="bar", color = "lightgreen")
plt.title("Binary Contribution to Model")
plt.show()

y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, color = "green")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Binary Features')
plt.show()

binaryd = scaler.transform(binary_d)

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, binaryd, binary_target, cv = 5,scoring = "roc_auc")
# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(np.mean(cv_auc)))



#k-Means clustering
from sklearn.cluster import KMeans
binary_data = pd.get_dummies(adult)
binary_data["income"] = binary_data["income_>50K"]
del binary_data["income_<=50K"]
del binary_data["income_>50K"]
binary_target = binary_data["income"].values
binary_d = binary_data.drop("income", axis = 1).values                       
binary_d = scaler.fit_transform(binary_d)
clust = KMeans(n_clusters=2)
clust.fit(binary_d)
labels = clust.predict(binary_d)
df = pd.DataFrame({'labels': labels, 'varieties': binary_target})
ct = pd.crosstab(df["labels"], df["varieties"])
cm = metrics.confusion_matrix(binary_target, labels)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["income"].classes_, yticklabels=encoders["income"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("Confusion Matrix of k-Means Clustering")
plt.show()
print ("F1 score: %f" % metrics.f1_score(binary_target, labels))
metrics.cohen_kappa_score(binary_target, labels)



#Random forest NOT USED
#Bagging
tree_count = 10 
bag_proportion = 0.6 
predictions = []
encoded_data, encoders = number_encode_features(adult)
adult_target = encoded_data["income"].values
adult_data = encoded_data.drop("income", axis = 1).values
adult_data1 = encoded_data.drop("income", axis = 1)            
X_train, X_test, y_train, y_test = train_test_split(adult_data, adult_target, test_size = 0.2, random_state = 15, stratify=adult_target)
#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
category_col =["age", "workclass", "fnlwgt", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
sss = StratifiedShuffleSplit(adult_target, 1, test_size=0.2, random_state=15) 
for train_index, test_index in sss:
    train_data = encoded_data.iloc[train_index] 
    test_data = encoded_data.iloc[test_index]
    
    for i in range(tree_count):
        bag = train_data.sample(frac=bag_proportion, replace = True, random_state=i)
        X_train, X_test = bag[category_col], encoded_data[category_col]
        y_train, y_test = bag["income"], encoded_data["income"]
        destree = DecisionTreeClassifier(random_state=1, min_samples_leaf=75) 
        destree.fit(X_train, y_train) 
        predictions.append(destree.predict_proba(X_test)[:,1])

combined = np.sum(predictions, axis=0)/10 
rounded= np.round(combined)
    
print(accuracy_score(rounded, y_test)) 
print(roc_auc_score(rounded, y_test))
#--------------------

from sklearn.ensemble import RandomForestClassifier
rndmfr = RandomForestClassifier(random_state=15)
encoded_data, encoders = number_encode_features(adult)
adult_target = encoded_data["income"].values
adult_data = encoded_data.drop("income", axis = 1).values
adult_data1 = encoded_data.drop("income", axis = 1)            
X_train, X_test, y_train, y_test = train_test_split(adult_data, adult_target, test_size = 0.2, random_state = 15, stratify=adult_target)
#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
rndmfr.fit(X_train, y_train)
pred = rndmfr.predict(X_test)
cv_scores = cross_val_score(rndmfr, adult_data, adult_target, cv = 5)
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
cm = metrics.confusion_matrix(y_test, pred)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["income"].classes_, yticklabels=encoders["income"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("Confusion Matrix of Random Forest")
plt.show()
print ("F1 score: %f" % metrics.f1_score(y_test, pred))
metrics.cohen_kappa_score(y_test, pred)
from sklearn.metrics import roc_curve
y_pred_prob = rndmfr.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, color = "green")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Binary Features')
plt.show()
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
# Compute predicted probabilities: y_pred_prob
y_pred_prob = rndmfr.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
    # Compute cross-validated AUC scores: cv_auc
    cv_auc = cross_val_score(rndmfr, adult_data, adult_target, cv = 5,scoring = "roc_auc")
    # Print list of AUC scores
    print("AUC scores computed using 5-fold cross-validation: {}".format(np.mean(cv_auc)))
list(zip(adult_data1, rndmfr.feature_importances_))