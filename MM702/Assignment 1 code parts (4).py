# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 08:47:29 2017

@author: HP
"""
#import the required packages for the analysis
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
from scipy.stats import pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#import the abalone file and add the column names
file = "abalone.data.csv"
os.chdir("D:/University of Brighton/2016-2017 Data Analytics MSc/2016 MM702 - Data Mining and Knowledge Discovery in Data OPTIONAL SEM 2 20CR/Assignment 1 Dataset Case Studies")
column_name = ["Sex", "Length", "Diameter", "Height", "Whole.Weight", "Shucked.Weight", "Viscera.Weight", "Shell.Weight", "Rings"]
abalone = pd.read_csv(file, names = column_name)
abalone.dtypes
abalone.head()
abalone.describe()

#drop the outlier
abalone = abalone.drop(abalone.index[[2051,1417]])

#draw up the histogram
fig = plt.figure(figsize=(18,13))
cols = 3
rows = (float(abalone.shape[1]) / cols)
for i, column in enumerate(abalone.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if abalone.dtypes[column] == np.object:
        abalone[column].value_counts().plot(kind="bar", axes=ax, color = "lightgreen")
    else:
        abalone[column].hist(axes=ax, color = "lightgreen")
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.3, wspace=0.2)

sns.pairplot(abalone, hue = "Rings")

#outlier identification
plt.scatter(abalone["Height"], abalone["Diameter"], color = "lightgreen")
plt.title("Outlier identification")
plt.show()
    
#heatmap of correlations
data, _ = number_encode_features(abalone)
mask = np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(data.corr(), mask = mask, square=True, cmap="RdYlGn")
ax.set_title("Abalone Heatmap")
plt.savefig("heatmap.png")
plt.show()

#The Sex column is type object which would confuse the kNN, thus we code dummy variables
abalone = pd.get_dummies(abalone, drop_first = True)

#normalize the data
abalone_normal = preprocessing.scale(abalone)

#separate the data to target and features
abalone_target = abalone["Rings"].values
abalone_data = abalone.drop("Rings", axis = 1).values
t_train, t_test, d_train, d_test = train_test_split(abalone_data, abalone_target, random_state = 15, test_size = 0.2)
abalone_target = abalone_target.reshape(-1,1)
scaler = preprocessing.StandardScaler()
t_train = scaler.fit_transform(t_train)
t_test = scaler.transform(t_test)


neighbors = np.arange(1, 40)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(t_train, d_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(t_train, d_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(t_test, d_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1,40)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv = 5)
knn_cv.fit(t_train, d_train)
knn_cv.best_params_
knn_cv.best_score_

#specify the kNN parameters
knn = KNeighborsClassifier(n_neighbors = 31)
cv_scores = cross_val_score(knn, t_train, d_train, cv = 5)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
knn.fit(t_train, d_train)
pred = knn.predict(t_test)
print(knn.score(t_test, d_test))
print(metrics.classification_report(d_test, pred))
print(metrics.confusion_matrix(d_test, pred))

#add volume
abalone['Volume'] = abalone['Height']*abalone['Length']*abalone['Diameter']
abalone = abalone.drop("Height", axis = 1)
abalone = abalone.drop("Diameter", axis = 1)
abalone = abalone.drop("Length", axis = 1)
abalone = pd.get_dummies(abalone, drop_first = True)


#separate the data to target and features
abalone_target = abalone["Rings"].values
abalone_data = abalone.drop("Rings", axis = 1).values
t_train, t_test, d_train, d_test = train_test_split(abalone_data, abalone_target, random_state = 15, test_size = 0.2)
abalone_target = abalone_target.reshape(-1,1)
scaler = preprocessing.StandardScaler()
t_train = scaler.fit_transform(t_train)
t_test = scaler.transform(t_test)


neighbors = np.arange(1, 40)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(t_train, d_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(t_train, d_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(t_test, d_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


#Decision tree
model = DecisionTreeRegressor(random_state = 1)
# fit the estimator to the data using CV
cv_scores_d = cross_val_score(model, t_train, d_train, cv = 5)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_d)))
# apply the model to the test and training data
model.fit(t_train, d_train)
predicted_test_y = model.predict(t_test)
predicted_train_y = model.predict(t_train)
def scatter_y(true_y, predicted_y):
    """Scatter-plot the predicted vs true number of rings
        Plots:
       * predicted vs true number of rings
       * perfect agreement line
       * +2/-2 number dotted lines
    Returns the root mean square of the error
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(true_y, predicted_y, '.k', color = "lightgreen")
    ax.plot([0, 30], [0, 30], '--k')
    ax.plot([0, 30], [2, 32], ':k')
    ax.plot([2, 32], [0, 30], ':k')
    rms = (true_y - predicted_y).std()
    ax.text(25, 3,
            "Root Mean Square Error = %.2g" % rms,
            ha='right', va='bottom')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xlabel('Rings Actual')
    ax.set_ylabel('Rings Predicted')
    return rms
scatter_y(d_train, predicted_train_y)
plt.title("Training data")
scatter_y(d_test, predicted_test_y)
plt.title("Test data")

#Max depth = 10         
model = DecisionTreeRegressor(max_depth=10, random_state = 1)
# fit the estimator to the data using cross validation
cv_scores_d = cross_val_score(model, t_train, d_train, cv = 5)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_d)))
# fit the estimator to the data
model.fit(t_train, d_train)
# apply the model to the test and train data
predicted_test_y = model.predict(t_test)
predicted_train_y = model.predict(t_train)
scatter_y(d_train, predicted_train_y)
plt.title("Training data")
rms_decision_tree = scatter_y(d_test, predicted_test_y)
plt.title("Test data")

data_percentage_array = np.linspace(10, 100, 10)
train_error = []
test_error = []
for data_percentage in data_percentage_array:
    model = DecisionTreeRegressor(max_depth=10)
    number_of_samples = int(data_percentage / 100. * len(d_train))
    model.fit(t_train[:number_of_samples,:], d_train[:number_of_samples])

    predicted_train_y = model.predict(t_train)
    predicted_test_y = model.predict(t_test)

    train_error.append((predicted_train_y - d_train).std())
    test_error.append((predicted_test_y - d_test).std())
plt.plot(data_percentage_array, train_error, label='training')
plt.plot(data_percentage_array, test_error, label='validation', )
plt.legend(loc=3)
plt.xlabel('Data percentage')
plt.ylabel('Root mean square error')
plt.title("Regression Tree Learning Curve")
plt.show()


#Unused visualisations
#EDA of physical measurements
vars=["Length", "Diameter", "Height", "Rings"]
sns.pairplot(abalone, vars=vars, hue="Sex")
#EDA of weight measurements
vars=["Whole.Weight", "Shucked.Weight", "Viscera.Weight",
      "Shell.Weight", "Rings"]
sns.pairplot(abalone, vars=vars, hue="Sex")
#EDA sex
g = sns.FacetGrid(abalone, col="Sex", margin_titles=True)
g.map(sns.regplot, "Whole.Weight", "Rings",
      fit_reg=False, x_jitter=.1);


#Adult dataset----------------------------------------------------------------------
file = "adult.data.csv"
column_name = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "gross_income"]
adult = pd.read_csv(file, names = column_name, index_col=False, sep=',\s', na_values=["?"])
adult.shape
adult.head()
#missing values proportions
adult.isnull().sum()/len(adult)*100
#clean missing values
adult = pd.DataFrame.dropna(adult)
adult.shape

sns.pairplot(adult, hue = "gross_income")

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

#Unsused correlation calculation
category_col =["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "gross_income"] 
adult.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['not married','married','married','married',
              'not married','not married','not married'], inplace = True)
for col in category_col:
    b, c = np.unique(adult[col], return_inverse=True) 
    adult[col] = c

#identify best parameters
col_names = adult.columns
param=[]
correlation=[]
abs_corr=[]
for c in col_names:
    #Check if binary or continuous
    if c != "gross_income":
        if len(adult[c].unique()) <= 2:
            corr = spearmanr(adult['gross_income'],adult[c])[0]
        else:
            corr = pointbiserialr(adult['gross_income'],adult[c])[0]
        param.append(c)
        correlation.append(corr)
        abs_corr.append(abs(corr))
#Create dataframe for visualization
param_df=pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr})
param_df=param_df.sort_values(by=['abs_corr'], ascending=False)
#Set parameter name as index
param_df=param_df.set_index('parameter')
param_df
best_features=param_df.index[0:4].values
print('Best features:\t',best_features)

#delete education as it represents the same as education-num
del adult["education"]
encoded_data, encoders = number_encode_features(adult)
adult_target = encoded_data["gross_income"].values
adult_data = encoded_data.drop("gross_income", axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(adult_data, adult_target, test_size = 0.2, random_state = 15, stratify=adult_target)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#LOGISTIC REG
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred),cm)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["gross_income"].classes_, yticklabels=encoders["gross_income"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("Confusion Matrix")
print ("F1 score: %f" % metrics.f1_score(y_test, y_pred))
plt.show()
logreg.score(y_test, y_pred)

from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
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
cv_auc = cross_val_score(logreg, adult_data, adult_target, cv = 5,scoring = "roc_auc")
# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

binary_data = pd.get_dummies(adult)
binary_data["gross_income"] = binary_data["gross_income_>50K"]
del binary_data["gross_income_<=50K"]
del binary_data["gross_income_>50K"]
binary_target = binary_data["gross_income"].values
binary_d = binary_data.drop("gross_income", axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(binary_d, binary_target, train_size=0.80)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred),cm)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["gross_income"].classes_, yticklabels=encoders["gross_income"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("Confusion Matrix of Binary Features")
plt.show()
print ("F1 score: %f" % metrics.f1_score(y_test, y_pred))

y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Generate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Binary Features')
plt.show()

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, adult_data, adult_target, cv = 5,scoring = "roc_auc")
# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

#k-Means
from sklearn.cluster import KMeans
binary_d = scaler.fit_transform(binary_d)
clust = KMeans(n_clusters=2)
clust.fit(binary_d)
labels = clust.predict(binary_d)
df = pd.DataFrame({'labels': labels, 'varieties': binary_target})
ct = pd.crosstab(df["labels"], df["varieties"])
print(ct)
cm = metrics.confusion_matrix(binary_target, labels)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["gross_income"].classes_, yticklabels=encoders["gross_income"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("Confusion Matrix of k-Means Clustering")
plt.show()
print ("F1 score: %f" % metrics.f1_score(binary_target, labels))


predictors = ['marital-status', 'education-num', 'relationship', 'age']
preds_d = encoded_data[predictors]
preds_t = encoded_data["gross_income"]
preds_d = scaler.fit_transform(preds_d)
clust = KMeans(n_clusters=2)
clust.fit(preds_d)
labels = clust.predict(preds_d)
df = pd.DataFrame({'labels': labels, 'varieties': preds_t})
ct = pd.crosstab(df["labels"], df["varieties"])
print(ct)
cm = metrics.confusion_matrix(binary_target, labels)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["gross_income"].classes_, yticklabels=encoders["gross_income"].classes_, cmap = "Greens")
plt.ylabel("Real value")
plt.xlabel("Predicted value")
plt.title("Confusion Matrix of Optimised k-Means Clustering")
plt.show()
print ("F1 score: %f" % metrics.f1_score(binary_target, labels))