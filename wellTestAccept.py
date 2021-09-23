# -*- coding: utf-8 -*-
"""
Model for welltest analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


#%% Read in the datasets

df = pd.read_csv('H:\Data_Science\welltestdata.csv')
dfpred = pd.read_csv('H:\Data_Science\predwelltestdata.csv')

df_clean = df

# Clean the Dataset
#well = 'N9N'
infills = 'N'
SAGDs = ('P','R')

df_clean = df_clean[np.isfinite(df_clean['Watercut BSW %'])]
#Keep Specific Wells
#df_clean = df_clean[df_clean['Well'] == well]

#Keep Only Infills
#df_clean = df_clean[df_clean['Well'].str.endswith(infills)]

#Keep only SAGD Wells
df_clean = df_clean[df_clean['Well'].str.endswith(SAGDs)]

df_clean = df_clean[df_clean['Test Status'] != 'NEW']
df_clean['Test Status'].replace({'ACCEPTED':1, 'REJECTED':0}, inplace = True)
df_clean.reset_index(drop = True, inplace = True)

# Select parameters

cdf = df_clean[['Test Status', 'Emulsion Temp degC', 'Liquid Density kg/m3', 'Operations BSW %', 'AGAR BSW %', 'Watercut BSW %', 'Test Emulsion Rate m3/d']]

#%% Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

#%% Model Training & Set Up

features = ['Watercut BSW %', 'Test Emulsion Rate m3/d']
#features = ['Watercut BSW %']
target = ['Test Status']

X = cdf[features].values
y = cdf[target].values


#X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X,y)
LR = LogisticRegression(C=0.01, solver='lbfgs').fit(X,y)

xpred = dfpred[features].values

#%% Test best K Value
#from sklearn import metrics
#Ks = 10
#mean_acc = np.zeros((Ks-1))
#std_acc = np.zeros((Ks-1))
#ConfustionMx = [];
#for n in range(1,Ks):
    
    #Train Model and Predict  
#    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
#    yhat=neigh.predict(X_test)
#    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
#    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])


#%% Predictions

#Predict KNN


yhatKNN = neigh.predict(xpred)
yhatKNNprob = neigh.predict_proba(xpred)
dfpred['KNN Predicted Status'] = yhatKNN
dfpred['KNN Probability'] = yhatKNNprob[:,0]


modKNNPred = []
n=0
for i in yhatKNNprob[:,0]:
    if i < 0.1 or i > 0.9:
        modKNNPred.append(yhatKNN[n])
        n=n+1
    else:
        modKNNPred.append("Low Prediction Probability")
        n=n+1
        

dfpred['KNN Modified Prediction'] = modKNNPred
# Predic Logistic Regression

# Compute confusion matrix
#cnf_matrix = confusion_matrix(y_test, yhatLR, labels=[1,0])
#np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

xpredLR = dfpred[features].values
yhatLR = LR.predict(xpredLR)
dfpred['LR Predicted Status'] = yhatLR
yhatLRprob = LR.predict_proba(xpredLR)
dfpred['LR Probability'] = yhatLRprob[:,0]

modLRPred = []

n=0
for i in yhatLRprob[:,0]:
    if i < 0.1 or i > 0.9:
        modLRPred.append(yhatLR[n])
        n=n+1
    else:
        modLRPred.append("Low Prediction Probability")
        n=n+1
        

dfpred['LR Modified Prediction'] = modLRPred

#%% Plots

# Data Plot Accepted or Rejected
x_plot = cdf['Test Status']
y_plot = cdf['Liquid Density kg/m3']

x2_plot = cdf['Test Status']
y2_plot = cdf['Watercut BSW %']


plt.subplot(3,1,1)
plt.scatter(x_plot, y_plot, color='blue')
plt.xlabel("Accept or Reject")
plt.ylabel("Density")
plt.title('Data Points on Well: '+well, loc='center')
plt.annotate("Number of Data Points: "+str(cdf['Test Status'].value_counts()), xy=(1,1), xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='bottom')

plt.subplot(3,1,2)
plt.scatter(x2_plot, y2_plot, color='red')
plt.xlabel("Accept or Reject")
plt.ylabel("BSW")
plt.title('Data Points on Well: '+well, loc='center')
plt.annotate("Number of Data Points: "+str(cdf['Test Status'].value_counts()), xy=(1,1), xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='bottom')

# K Value Plot
plt.subplot(3,1,3)    
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.title('Accuracy of Model @ Different k Values', loc='center')
plt.annotate("The best accuracy was "+str(mean_acc.max())+" with k= "+str(mean_acc.argmax()+1), xy=(1,1), xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='bottom')
plt.tight_layout()
plt.show()

#%% Save to csv
import datetime
now = datetime.datetime.now()
#dfpred.to_csv('H:\Data_Science\results'+str(now.hour)+str(now.minute)+str(now.second)+'.csv')
dfpred.to_csv(r'H:\Data_Science\results'+str(now.hour)+str(now.minute)+str(now.second)+'.csv', index = None, header=True)
