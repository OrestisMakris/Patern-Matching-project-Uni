import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import mean
from numpy import std

Patient_data = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv" , header=None )


Patient_data[1].replace('Male' ,0,inplace=True)
Patient_data[1].replace('Female' ,1,inplace=True)
Patient_data[10].replace(1 ,0,inplace=True)
Patient_data[10].replace(2 ,1,inplace=True)

#for i in range(len(Patient_data)):
 #   print(Patient_data[1][i])

Patient_data=Patient_data.dropna()

#print(Patient_data.to_string())

#normalized_arr = preprocessing.normalize(Patient_data)

#scaler = MinMaxScaler(feature_range=(-1, 1))
#norma = scaler.fit_transform(Patient_data)

Patient_data[0] = 2*((Patient_data[0] - min(Patient_data[0])) / ( max(Patient_data[0]) - min(Patient_data[0]) ))-1
for i in range(2,10):
    Patient_data[i] = 2*((Patient_data[i] - min(Patient_data[i])) / ( max(Patient_data[i]) - min(Patient_data[i]) ))-1

print(Patient_data.to_string())

Patient_data_Y = Patient_data.iloc[:,10:]
Patient_data_X = Patient_data.iloc[:,:-1]


from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
Patient_data_X_train, Patient_data_X_test, Patient_data_Y_train, Patient_data_Y_test = train_test_split(Patient_data_X, Patient_data_Y, test_size=0.18,random_state=230) # 70% training and 30% test
#Patient_data_X_train, Patient_data_X_test, Patient_data_Y_train, Patient_data_Y_test = train_test_split(Patient_data_X, Patient_data_Y, test_size=0.3,random_state=235) # 70% training and 30% test

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(Patient_data_X_train, Patient_data_Y_train)

y_pred = classifier.predict(Patient_data_X_test)

from sklearn.metrics import confusion_matrix,accuracy_score

ac = accuracy_score(Patient_data_Y_test ,y_pred)
cm = confusion_matrix(Patient_data_Y_test , y_pred)

print (ac)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=1, shuffle=True)

# evaluate model
scores = cross_val_score(classifier, Patient_data_X, Patient_data_Y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print(scores)
    

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(Patient_data_X, Patient_data_Y)
y_pred1 = svclassifier.predict(Patient_data_X_test)
ac = accuracy_score(Patient_data_Y_test ,y_pred1)
print (ac)

#classifier = GaussianNB()
#classifier1.fit(Patient_data_X_train, Patient_data_Y_train)