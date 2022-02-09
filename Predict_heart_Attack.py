#import necessary libraries
import sys
import os
import warnings
warnings.filterwarnings('ignore')

#!{sys.executable} -m pip install numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns

#read current path
os.chdir(r"C:\Users\masou")
cwd = os.getcwd()
d='echocardiogram.data'
t='echocardiogram.test'
#find data in the current path
path=1
for dirpath, _, filenames in os.walk(cwd):
  if path:
    df= pd.read_csv(os.path.abspath(os.path.join(dirpath, d)),na_values=["?"],error_bad_lines=False,header=None)
    # test data has been modified since in line 16, we had 14 attribute rather than 13.

    test_data= pd.read_csv(os.path.abspath(os.path.join(dirpath, t)),na_values=["?"],error_bad_lines=False,header=None)

  path=0

# percentage of missing data per category
total = df.isnull().sum().sort_values(ascending=False)
percent_total = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100
missing = pd.concat([total, percent_total], axis=1, keys=["Total", "Percentage"])
missing_data = missing[missing['Total']>0]
print(missing_data)

plt.figure(figsize=(9,6))
sns.set(style="whitegrid")
sns.barplot(x=missing_data.index, y=missing_data['Percentage'], data = missing_data)
plt.title('Percentage of missing data by attributes')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.show()

# plot histogram to see the distribution of the data
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax = ax)
plt.show()

# predict whether a patient (has) survived at least 2 years.
#df= df[~(df.iloc[:,0] < 24) & df.iloc[:,1]]
#df= df[~(df.iloc[:,0] < 24)]
#numInstanc=len(df)

# Decide about patients with duration survival of at least 24 months
# if condition is satisfied 1, otherwise 0

dfLabel= (df.iloc[:,0]>=24).astype(int)
# Remove columns that don't provide any information
df.drop(df.columns[[0,1,9,10,11,12]], axis=1, inplace = True) #to drop desired columns
test_data.drop(test_data.columns[[0,1,9,10,11,12]], axis=1, inplace = True)

# to reset the index of columns
#df.reset_index(drop=True)
#test_data.reset_index(drop=True, inplace=True)

na_free = df.dropna()
#only_na = df[~df.index.isin(na_free.index)] # rows that have NA values

dfLabel=dfLabel[df.index.isin(na_free.index)]
df=df[df.index.isin(na_free.index)]

#dfLabel=dfLabel[~dfLabel.index.isin(only_na.index.values)]
#df = df.dropna(how='any',axis=0)
dfLabel.reset_index(drop=True, inplace=True) #We can use the drop parameter to avoid the old index being added as a column
df.reset_index(drop=True, inplace=True) #We can use the drop parameter to avoid the old index being added as a column

#df=df[df.iloc[:,[0,1,3]].notnull()] # find multiple na in one or columns and remove their accossiated rows
#test_data=test_data[test_data.iloc[:,0].notnull()] # to find the desired by solving column index issue- iloc

test_data = test_data.dropna(how='any',axis=0)
test_data.reset_index(drop=True, inplace=True)

# check if there is any duplicate row in the data
#df.groupby(df.columns.tolist(),as_index=False).size()
#test_data.groupby(test_data.columns.tolist(),as_index=False).size()


print(df,'\n',"number of instances in training is",len(df))
print(test_data,'\n',"number of instances in test is",len(test_data))


# split the dataset
X_train,X_test,y_train,y_test = train_test_split(df,dfLabel,test_size=.2,random_state=0)


#=============tune parameters===================
#1 using grid search for finding best parameters for logistic regression
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
#logisticRegr = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
#===================================================
#2 all parameters not specified are set to their defaults
logisticRegr = LogisticRegression(solver = 'lbfgs',penalty='l2')
#2 all parameters not specified are set to their defaults
#logisticRegr = LogisticRegression()
#train the classifier
logisticRegr.fit(X_train,y_train)

y_hat= logisticRegr.predict(X_test)

# Use score method to get accuracy of model
#accuracy is defined as:(fraction of correct predictions): correct predictions / total number of data points

score = logisticRegr.score(X_test, y_test)

logisticRegr_accuracy = accuracy_score(y_test,y_hat)
print(score)

cm = metrics.confusion_matrix(y_test, y_hat)
print(cm)

cm=confusion_matrix(y_test,y_hat)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

print(classification_report(y_test, y_hat))

logisticRegr_f1 = f1_score(y_test, y_hat)
print(f'The f1-score for logistic regression is {round(logisticRegr_f1*100,2)}%')

# see the result under ROC and AUC curve
logit_roc_auc = roc_auc_score(y_test, logisticRegr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logisticRegr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# run prediction with SVM
# train the model
#----------tune parameters----------------
#1 # defining parameter range
'''
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

# https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
clf = GridSearchCV(SVC(probability=True), param_grid, refit = True, verbose = 3)

# fitting the model for grid search
clf.fit(X_train, y_train)

#show best parameters
print(clf.best_estimator_)
print(clf.best_params_)
'''
#----------------------------------------------------------
#clf = SVC(kernel='linear',probability=True)
clf = SVC(kernel='rbf',probability=True)
clf.fit(X_train,y_train)

# predictions
y_hat_svm = clf.predict(X_test)


#accuracy
svm_accuracy = accuracy_score(y_test,y_hat_svm)
print(f"Using SVM we get an accuracy of {round(svm_accuracy*100,2)}%")

cm=confusion_matrix(y_test,y_hat_svm)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")

svm_f1 = f1_score(y_test, y_hat_svm)
print(f'The f1 score for SVM is {round(svm_f1*100,2)}%')

# see the result under ROC and AUC curve

svm_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr_svm, tpr_svm, label='SVM (AUC = %0.2f)' % svm_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('svm_ROC')
plt.show()

# model comparison
comparison = pd.DataFrame({
  "Logistic regression":{'Accuracy':logisticRegr_accuracy, 'AUC':logit_roc_auc, 'F1 score':logisticRegr_f1},
  "Support vector machine":{'Accuracy':svm_accuracy, 'AUC':svm_roc_auc, 'F1 score':svm_f1}
}).T
print("\n",comparison)


# cross-validation analysis for better model

cv_results = cross_val_score(clf, df, dfLabel, cv=10)
print("%0.2f accuracy with a standard deviation of %0.2f for SVM classifier" % (cv_results.mean(), cv_results.std()))

#------------test on unlabled data
y_unlabled= clf.predict(test_data)

print(y_unlabled)


