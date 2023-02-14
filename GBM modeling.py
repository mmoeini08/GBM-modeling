# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:20:51 2023

@author: mmoein2
"""

# %%Importing Libraries needed to run the tool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #split data in training and testing set
from sklearn.model_selection import cross_val_score #K-fold cross validation
from sklearn.preprocessing import StandardScaler #Scaling feature
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import time
from datetime import timedelta
start_time = time.monotonic()
sns.set(style='darkgrid')
data2=[]

# %%Read the file
data = pd.read_csv(".... .csv", header=0)

# %%Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")


# %% Removing rows without meaningful data points
data = data[data.wfh_pre != "Question not displayed to respondent"]
data = data[data.wfh_now != "Question not displayed to respondent"]
data = data[data.wfh_expect != "Question not displayed to respondent"]
data = data[data.jobcat_now_w1b != "Question not displayed to respondent"]
data = data[data.jobcat_now_w1b != "Variable not available in datasource"]

# %% MApping data into dummies
data['wfh_pre']= data['wfh_pre'].map({'Yes':1 ,'No' :0,'':0})
data['gender']= data['gender'].map({'Female':1 ,'Male':0,'':0})
data['wfh_now']= data['wfh_now'].map({'Yes':1 ,'No':0,'':0})

# %% Introducing data variables
X = np.column_stack((data.wfh_pre, data.wfh_now, data.age, data.gender,
                     pd.get_dummies(data.educ), pd.get_dummies(data.hhincome),
                     pd.get_dummies((data.jobcat_now_w1b))))
Y = data.wfh_expect

# %% Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)#, stratify=Y)

# %% Scaling data

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# %%
# %%
# %%
# %% Modeling
# %%
# %%


CV=5 #number of folds for crossvalidation

# %% Hyperparameter tuning

Accuracy=[]

for j in range (3, 11):
    print(j)
    for i in range (1,101):
        #gbm fit
        gbm = GradientBoostingClassifier(n_estimators=i, max_depth=j)
        #cross validation
        gbm_scores = cross_val_score(gbm, X_train, Y_train, cv=CV)
        Accuracy.append(gbm_scores.mean().round(2))
        
# %%        
# %% Plotting
# %% 
# %% 

# %% Hyperparamters tuning results (Figure1)   
fig1, axs = plt.subplots(2, 2, figsize=(20, 16))
axs[0, 0].plot(range(1, 100), Accuracy[1:100], color='blue')
axs[0, 0].axis('tight')
axs[0, 0].set_xlabel('Number of Estimators')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_title("max depth = 3")
axs[0, 0].set_ylim(0.6,0.9)


axs[0, 1].plot(range(1, 100), Accuracy[101:200], color='black')
axs[0, 1].axis('tight')
axs[0, 1].set_xlabel('Number of Estimators')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].set_title("max depth = 4")
axs[0, 1].set_ylim(0.6,0.9)


axs[1, 0].plot(range(1, 100), Accuracy[201:300], color='green')
axs[1, 0].axis('tight')
axs[1, 0].set_xlabel('Number of Estimators')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].set_title("max depth = 5")
axs[1, 0].set_ylim(0.6,0.9)


axs[1, 1].plot(range(1, 100), Accuracy[301:400], color='red')
axs[1, 1].axis('tight')
axs[1, 1].set_xlabel('Number of Estimators')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].set_title("max depth = 6")
axs[1, 1].set_ylim(0.6,0.9)

plt.show()
plt.savefig('GBM_Hyperparameter1_max_depth_Nestimators.png')
# %% Hyperparamters tuning results (Figure2)   

fig2, axs = plt.subplots(2, 2, figsize=(20, 16))
axs[0, 0].plot(range(1, 100), Accuracy[401:500], color='blue')
axs[0, 0].axis('tight')
axs[0, 0].set_xlabel('Number of Estimators')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_title("max depth = 7")
axs[0, 0].set_ylim(0.6,0.9)


axs[0, 1].plot(range(1, 100), Accuracy[501:600], color='black')
axs[0, 1].axis('tight')
axs[0, 1].set_xlabel('Number of Estimators')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].set_title("max depth = 8")
axs[0, 1].set_ylim(0.6,0.9)


axs[1, 0].plot(range(1, 100), Accuracy[601:700], color='green')
axs[1, 0].axis('tight')
axs[1, 0].set_xlabel('Number of Estimators')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].set_title("max depth = 9")
axs[1, 0].set_ylim(0.6,0.9)


axs[1, 1].plot(range(1, 100), Accuracy[701:800], color='red')
axs[1, 1].axis('tight')
axs[1, 1].set_xlabel('Number of Estimators')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].set_title("max depth = 10")
axs[1, 1].set_ylim(0.,0.9)

plt.show()
plt.savefig('GBM_Hyperparameter2_max_depth_Nestimators.jpeg')

# %% 
# %% 
# %% Using the best hyperparameters
# %% 
# %% Introducing the Model

gbm = GradientBoostingClassifier(n_estimators=15, max_depth=3)


# %% Cross Validation (CV) process
gbm_scores = cross_val_score(gbm, X_train, Y_train, cv=CV)
print("Gradient Boosting Accuracy: {0} (+/- {1})".format(gbm_scores.mean().round(2), (gbm_scores.std() * 2).round(2)))
print("")

# %% Training final algorithms
gbm = GradientBoostingClassifier(n_estimators=60, max_depth=10)
gbm_train=gbm.fit(X_train, Y_train) 


# %% Final Predictions
gbm_predict = gbm.predict(X_test)
gbm_score = metrics.accuracy_score(gbm_predict, Y_test)
#print("Gradient Boosting: {0}".format(gbm_predict))
print("Gradient Boosting score: {0}".format(gbm_score))
print("")

# %% 
# %% 
# %% Plotting the results
# %% 
# %% 

# %% Variable Importance Plots
nb_var = np.arange(len(X.columns))
plt.barh(nb_var, gbm.feature_importances_)
plt.yticks(nb_var, X.columns)
plt.title('Gradient Boosting Machine Feature Importance')
plt.tight_layout()
plt.savefig('GBM_Importance.JPEG') #Saving the plot
plt.show()
print("") 

# %% Confusion Matrix

confusion_matrix = metrics.confusion_matrix(gbm_predict, Y_test,normalize='true')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig('Confusion_GBM.JPEG') #Saving the plot
plt.show()


# %% Recording the time of runnig
time_duration=[]
end_time= time.monotonic()
time_duration.append(end_time - start_time)
print(time_duration)