import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import seaborn as sns
from numpy import mean
from numpy import std

#Importing Project data from the csv using Pandas dataframe
Patient_data = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv" , header=None )

#Using raplace function from Pandas dataframe to encode the gendrs with the numeric values 0 Male 1 Famale  
Patient_data[1].replace('Male' ,0,inplace=True)
Patient_data[1].replace('Female' ,1,inplace=True)

#Using raplace function from Pandas dataframe to replace the Clases Encoding with 0 = 1 , 1=2   
Patient_data[10].replace(1 ,0,inplace=True)
Patient_data[10].replace(2 ,1,inplace=True)

Patient_data=Patient_data.dropna()

fig,axes = plt.subplots(1 ,12 ,sharey=True)
sns.histplot(Patient_data , ax =axes[0] , x=0 , kde=True )
sns.histplot(Patient_data , ax =axes[1] , x=1 , kde=True )
sns.histplot(Patient_data , ax =axes[2] , x=2 , kde=True , color = 'b')
sns.histplot(Patient_data , ax =axes[3] , x=3 , kde=True , color = 'g')
sns.histplot(Patient_data , ax =axes[4] , x=4 , kde=True , color = 'c')
sns.histplot(Patient_data , ax =axes[5] , x=5 , kde=True )
sns.histplot(Patient_data , ax =axes[6] , x=6 , kde=True ,color = 'g')
sns.histplot(Patient_data , ax =axes[7] , x=7 , kde=True ,color = 'g')
sns.histplot(Patient_data , ax =axes[8] , x=8 , kde=True , color = 'b')
sns.histplot(Patient_data , ax =axes[9] , x=9 , kde=True , color = 'g')
sns.histplot(Patient_data , ax =axes[10] , x=10, kde=True , color = 'c')

Patient_data[0] = np.log(Patient_data[0])

for i in range(2,10):
   Patient_data[i] = np.log(Patient_data[i])



fig,axes = plt.subplots(1 ,12 ,sharey=True)
sns.histplot(Patient_data , ax =axes[0] , x=0 , kde=True )
sns.histplot(Patient_data , ax =axes[1] , x=1 , kde=True )
sns.histplot(Patient_data , ax =axes[2] , x=2 , kde=True , color = 'b')
sns.histplot(Patient_data , ax =axes[3] , x=3 , kde=True , color = 'g')
sns.histplot(Patient_data , ax =axes[4] , x=4 , kde=True , color = 'c')
sns.histplot(Patient_data , ax =axes[5] , x=5 , kde=True )
sns.histplot(Patient_data , ax =axes[6] , x=6 , kde=True ,color = 'g')
sns.histplot(Patient_data , ax =axes[7] , x=7 , kde=True ,color = 'g')
sns.histplot(Patient_data , ax =axes[8] , x=8 , kde=True , color = 'b')
sns.histplot(Patient_data , ax =axes[9] , x=9 , kde=True , color = 'g')
sns.histplot(Patient_data , ax =axes[10] , x=10, kde=True , color = 'c')


corr = Patient_data.iloc[:,:-1].corr(method='pearson')
cmap = sns.diverging_palette(250,354,80,60 , center='dark' , as_cmap=True)
sns.heatmap(corr , vmax =1 , vmin =.3 , cmap=cmap ,  annot=True,square =True , linewidths = .2)


Patient_data[0] = 2*((Patient_data[0] - min(Patient_data[0])) / ( max(Patient_data[0]) - min(Patient_data[0]) ))-1
for i in range(2,10):
    Patient_data[i] = 2*((Patient_data[i] - min(Patient_data[i])) / ( max(Patient_data[i]) - min(Patient_data[i]) ))-1

fig,axes = plt.subplots(1 ,11 ,sharey=True)
sns.histplot(Patient_data , ax =axes[0] , x=0 , kde=True )
sns.histplot(Patient_data , ax =axes[1] , x=1 , kde=True )
sns.histplot(Patient_data , ax =axes[2] , x=2 , kde=True , color = 'b')
sns.histplot(Patient_data , ax =axes[3] , x=3 , kde=True , color = 'g')
sns.histplot(Patient_data , ax =axes[4] , x=4 , kde=True , color = 'c')
sns.histplot(Patient_data , ax =axes[5] , x=5 , kde=True )
sns.histplot(Patient_data , ax =axes[6] , x=6 , kde=True ,color = 'g')
sns.histplot(Patient_data , ax =axes[7] , x=7 , kde=True ,color = 'g')
sns.histplot(Patient_data , ax =axes[8] , x=8 , kde=True , color = 'b')
sns.histplot(Patient_data , ax =axes[9] , x=9 , kde=True , color = 'g')
sns.histplot(Patient_data , ax =axes[10] , x=10, kde=True , color = 'c')
plt.show()


Patient_data_Y = Patient_data.iloc[:,10:]
Patient_data_X = Patient_data.iloc[:,:-1]

# Split dataset into training set and test set
Patient_data_X_train, Patient_data_X_test, Patient_data_Y_train, Patient_data_Y_test = train_test_split(Patient_data_X, Patient_data_Y, test_size=0.2,random_state=210) # 70% training and 30% test

classifier = GaussianNB()
classifier.fit(Patient_data_X_train, Patient_data_Y_train)

y_pred = classifier.predict(Patient_data_X_test)

ac = accuracy_score(Patient_data_Y_test ,y_pred)
cm = confusion_matrix(Patient_data_Y_test , y_pred)

print (ac)
print(cm)

cv = KFold(n_splits=5, random_state=1, shuffle=True)

# evaluate model
scores = cross_val_score(classifier, Patient_data_X, Patient_data_Y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print(scores)

#-------------------------------------------------------

Patient_data_X_rd = Patient_data_X[[0,1,4 ]]
#Patient_data_X_rd = Patient_data_X[[0,1]]
Patient_data_X_train_rd, Patient_data_X_test_rd, Patient_data_Y_train_rd, Patient_data_Y_test_rd = train_test_split(Patient_data_X_rd, Patient_data_Y, test_size=0.2,random_state=218)

classifier_reduced_dependency =  GaussianNB()
classifier_reduced_dependency.fit(Patient_data_X_train_rd, Patient_data_Y_train_rd)

y_pred_rd = classifier_reduced_dependency.predict(Patient_data_X_test_rd)

ac_rd = accuracy_score(Patient_data_Y_test_rd ,y_pred_rd)
cm_rd = confusion_matrix(Patient_data_Y_test_rd, y_pred_rd)

print (ac_rd)
print(cm_rd)

cv_rd = KFold(n_splits=5, random_state=1, shuffle=True)

# evaluate model
scores_rd = cross_val_score(classifier_reduced_dependency, Patient_data_X_rd, Patient_data_Y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print(scores_rd)