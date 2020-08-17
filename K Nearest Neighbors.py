# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:22:39 2020

@author: Dell
"""

# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Importing the dataset
dataset = pd.read_csv('data.csv')

# Exploring the dataset
dataset.head()

# Standardizing the variables
scaler = StandardScaler()
scaler.fit(dataset.drop('TARGET CLASS',axis=1))

# Transforming feature to scaled version
scaled_features = scaler.transform(dataset.drop('TARGET CLASS',axis=1))

# Checking if the scaling worked
dataset_featured = pd.DataFrame(scaled_features,columns=dataset.columns[:-1])
dataset_featured.head()

# Spliting the dataset in training and testing set
X_train, X_test, y_train, y_test = train_test_split(scaled_features,dataset['TARGET CLASS'], test_size=0.30)

# Using KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

# Evaluating the prediction
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Chossing a good value of K
error_rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# From the plot it is evident that the error rate is minimum for the K = 34 value so now
# using this value of K = 34 we will test and train the our model again

knn = KNeighborsClassifier(n_neighbors=34)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
