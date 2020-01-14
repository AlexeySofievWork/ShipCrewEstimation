#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:36:44 2020

@author: as
"""

import pandas
from sklearn.svm import SVR # Support Vector Machine, Regression
import matplotlib.pyplot as plt
import seaborn as sns  # for heatmap
import numpy as np
from sklearn.model_selection import train_test_split  # For cross validation
from scipy.stats import pearsonr  # Pearson correlation


## 1. Read the file and display columns.
dataset=pandas.read_csv("cruise_ship_info.csv")
#print dataset

## 2. Calculate basic statistics of the data (count, mean, std, etc) and examine data and state your observations.
#%%
print "Shape is: ", dataset.shape
print "Columns are: ", dataset.columns

#Todo, add some more staff here...

print "Crew average number: ", dataset.iloc[:,-1].mean()
print "Crew std: ", dataset.iloc[:,-1].std()
## Etc... Unsure why needed, but lets printed.

## Now the actual thing! Heatmap illustrates the correspondence between the classes!!!

fig, ax = plt.subplots(figsize=(9,6))
sns.heatmap(dataset.corr(), center=0, cmap='Blues')

print "From the heatmap above we see that there is relation between other classes, except age and passenger density."
print "This might be meaningful result when optimizing the model."

## 3. Select columns that will be probably important to predict “crew” size.
## 4. If you removed columns explain why you removed those.
#Answer: Age and passenger density are probably unneeded columns, but will not drop anything out. 
#Answer: SVM can do it itself, and the SVR will predict the results based on meaningful ones.10000000

## 5. Use one-hot encoding for categorical features.
#%%
print("Original features:\n", list(dataset.columns), "\n")
data_dummies = pandas.get_dummies(dataset)
print("Features after get_dummies:\n", list(data_dummies.columns))


## 6. Create training and testing sets (use 60% of the data for the training and reminder for testing).
#%%
#X=dataset.loc[:, dataset.columns != 'Crew' and dataset.columns != 'Ship_name' and dataset.columns != 'Cruise_line'] # All data except Crew size and names
X=dataset.loc[:, ~dataset.columns.isin(['crew','Ship_name', 'Cruise_line']) ] # All data except Crew size and names
Y=dataset.loc[:, dataset.columns.isin(['crew'])] # Answer we want to get
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=42) # rng seed=42
print "X_train, X_test, y_train, y_test:", X_train.shape, X_test.shape, y_train.shape, y_test.shape



## 7. Build a machine learning model to predict the ‘crew’ size.
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)
print("Rbf kernel Test score: {:.2f}".format(regressor.score(X_test, y_test)))

# 0.15 not good, lets try the basic linear kernel.

regressor = SVR(kernel='linear')
regressor.fit(X_train,y_train)
print("Linear kernel Test score: {:.2f}".format(regressor.score(X_test, y_test)))

#Score 0.94 is pretty good! :)


## 8. Calculate the Pearson correlation coefficient for the training set and testing data sets.
#%%

# Lets use the easy way, scipy's pearsonr. 

regressor = SVR(kernel='linear',C=1)
regressor.fit(X_train,y_train)
y_prediction=regressor.predict(X_test)
corr, _ = pearsonr(y_prediction, y_test.crew.to_numpy())
print('Pearsons correlation: %.3f' % corr)

## 9. Describe hyper-parameters in your model and how you would change them to improve the performance of the model.
#%%

# From Source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

"""
kernelstring, optional (default=’rbf’)

    Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.



Cfloat, optional (default=1.0)

    Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.


epsilonfloat, optional (default=0.1)

    Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.


"""

# Practically there are 3 things to optimize: 1) kernel, (Changed RBF to linear since it works better)
# 2) Regularization parameter. Can be used to control overlearning. 0.94 is good result, so not worth optimizing
# 3) epsilon: the tolerance parameter.
# Pretty often the epsilon of 0.2 is used instead of default 0.1


regressor = SVR(kernel='linear', C=1, epsilon=0.2)
regressor.fit(X_train,y_train)
print("Linear kernel Test score with epsilon 0.2: {:.4f}".format(regressor.score(X_test, y_test)))


regressor = SVR(kernel='linear', C=1, epsilon=0.1)
regressor.fit(X_train,y_train)
print("Linear kernel Test score: {:.4f}".format(regressor.score(X_test, y_test)))

# As it can be seen -- there is slight improvement. 



## 10.What is regularization? What is the regularization parameter in your model?

# Regularization is technique to prevent overfitting model to data. In SVR it is reg.par. C. Default: 1.0

## 11.Plot regularization parameter value vs Pearson correlation for the test and training sets, and see whether your model has bias problem or variance problem.

#%% Lets calculate with regularization step of 0.1 the predictions and PearsonCorrelations to them
#!!! NOTE !!! this is slow, since model have to be retrained every time...

a2CvsPearson=np.zeros([100,2]);
for i in range(1,100):
    if i%5==0:
        print "Regularization vs Pearson plot is ", i, "% calculated."
    dTmp=i/10.0;
    regressor = SVR(kernel='linear',C=dTmp)
    regressor.fit(X_train,y_train)
    y_prediction=regressor.predict(X_test)
    corr, _ = pearsonr(y_prediction, y_test.crew.to_numpy())
    #print('Pearsons correlation: %.3f' % corr)
    a2CvsPearson[i,0]=dTmp;
    a2CvsPearson[i,1]=corr;
    

#%%  plotting
fig, ax = plt.subplots()
ax.plot(a2CvsPearson[1:,0],a2CvsPearson[1:,1],'*') #skipping first value since regularization 0 doesn't exist

ax.set(xlabel='Regularization parameter', ylabel='Pearson correlation coefficient',
       title='Linear SVR, regularization vs Pearson correlation')
plt.show()

# Conclusion: strong correlation between predictions and real data --> predictions are most likely to be correct.
# Variation in P.corr depending on regularition is small -- no clear bias / errors.


