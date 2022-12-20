#Importing the neceserie Librarys 
#Numpy for the math Oparations  
import numpy as np
#Pandas for the data preparatuon
import pandas as pd
#Matplotlib and seabron for the ploting of the data
import matplotlib.pyplot as plt
import seaborn as sns
#skit_learn library from model_selection pacjage imporitng train_test_split in order to the arrays or matrices into random train and test subsets.
from sklearn.model_selection import train_test_split
#skit_learn library from naive_bayes pacjage in order to use the GaussianNB naive bayes clasifiaer 
from sklearn.naive_bayes import GaussianNB
#skit_learn library from metrics pacjage in order to use the confusion_matrix,accuracy_score and ConfusionMatrixDisplay for avalting our model 
from sklearn.metrics import confusion_matrix,accuracy_score , ConfusionMatrixDisplay\
#skit_learn library from model_selection pacjage imporitng cross_val_score in order to use the k-fold cross validation method to avalute our model 
from sklearn.model_selection import cross_val_score


#Importing Project data from the csv using Pandas dataframe
Patient_data = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv" , header=None )

#Using raplace function from Pandas dataframe to encode the gendrs with the numeric values 0 Male 1 Famale  
Patient_data[1].replace('Male' ,0,inplace=True)
Patient_data[1].replace('Female' ,1,inplace=True)

#Using raplace function from Pandas dataframe to replace the Clases Encoding it with 0=1 , 1=2   
Patient_data[10].replace(1 ,0,inplace=True)
Patient_data[10].replace(2 ,1,inplace=True)

#Displaying all the data from the Pandas dataframe using to_string() method
print(Patient_data.to_string())

#Using dropna method from Pandas dataframe to remove all the rows that contains NULL or NAN values.
Patient_data=Patient_data.dropna()

#Using value_counts method from Pandas dataframe to return an object with counts of unique values.
print(Patient_data[10].value_counts())

#Using hist method from Pandas dataframe to display a histogram a representation of the distribution of data for the lsast calum of the data frame wich cointains the classes
Patient_data[10].hist(color = "darkCyan")
plt.show()

#Using seaborn edn matplotlib librairies we plot 10 histograms one per colmn (feature)
fig,axes = plt.subplots(1 ,10 ,figsize=(60,15),sharey=True)
sns.histplot(Patient_data , ax =axes[0] , x=0 , kde=True , color = '#a1c9f4').set(title='Age')
sns.histplot(Patient_data , ax =axes[1] , x=1, kde=True , color = '#8de5a1').set(title='Gende')
sns.histplot(Patient_data , ax =axes[2] , x=2 , kde=True , color =  '#B0171F').set(title='TB')
sns.histplot(Patient_data , ax =axes[3] , x=3 , kde=True, color = '#d0bbff' ).set(title='DB')
sns.histplot(Patient_data , ax =axes[4] , x=4 , kde=True , color = '#D02090').set(title='Alkphos Alkaline Phosphotase')
sns.histplot(Patient_data , ax =axes[5] , x=5 , kde=True , color = '#1C86EE').set(title='Sgpt')
sns.histplot(Patient_data , ax =axes[6] , x=6 , kde=True, color = '#00688B').set(title='Sgot')
sns.histplot(Patient_data , ax =axes[7] , x=7 , kde=True , color = '#EE7600').set(title='TP')
sns.histplot(Patient_data , ax =axes[8] , x=8 , kde=True, color = '#8E388E').set(title='ALB')
sns.histplot(Patient_data , ax =axes[9] , x=9 , kde=True , color = '#CD2626') .set(title='A/G')
plt.show()


#The log() numpy method a mathematical function that calculates natural logarithm of echa allmento of the array here for the first feature
Patient_data[0] = np.log(Patient_data[0])

#The log() numpy method a mathematical function that calculates natural logarithm of echa allmento of the array here for remainig fetures expect the gender
for i in range(2,10):
   Patient_data[i] = np.log(Patient_data[i])


#Using seaborn edn matplotlib librairies we plot again the historagram in order to see the resuts of the logarithms 
fig,axes = plt.subplots(1 ,10 ,figsize=(60,15),sharey=True)
sns.histplot(Patient_data , ax =axes[0] , x=0 , kde=True , color = '#a1c9f4').set(title='Age')
sns.histplot(Patient_data , ax =axes[1] , x=1, kde=True , color = '#8de5a1').set(title='Gende')
sns.histplot(Patient_data , ax =axes[2] , x=2 , kde=True , color =  '#B0171F').set(title='TB')
sns.histplot(Patient_data , ax =axes[3] , x=3 , kde=True, color = '#d0bbff' ).set(title='DB')
sns.histplot(Patient_data , ax =axes[4] , x=4 , kde=True , color = '#D02090').set(title='Alkphos Alkaline Phosphotase')
sns.histplot(Patient_data , ax =axes[5] , x=5 , kde=True , color = '#1C86EE').set(title='Sgpt')
sns.histplot(Patient_data , ax =axes[6] , x=6 , kde=True, color = '#00688B').set(title='Sgot')
sns.histplot(Patient_data , ax =axes[7] , x=7 , kde=True , color = '#EE7600').set(title='TP')
sns.histplot(Patient_data , ax =axes[8] , x=8 , kde=True, color = '#8E388E').set(title='ALB')
sns.histplot(Patient_data , ax =axes[9] , x=9 , kde=True , color = '#CD2626') .set(title='A/G')
plt.show()

#using seaborn heatmap method in order to plot the heatmap of the feutres in order to see the dipendaticy of each feuter with the other 
corr = Patient_data.iloc[:,:-1].corr(method='pearson')
cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
sns.heatmap(corr , vmax =1 , vmin =.3 , cmap=cmap ,  annot=True,square =True , linewidths = .2)
plt.show()

#Normalise the data of each feature betweeen the -1 and 1 expect the gender (column with index 1)
Patient_data[0] = 2*((Patient_data[0] - min(Patient_data[0])) / ( max(Patient_data[0]) - min(Patient_data[0]) ))-1
for i in range(2,10):
    Patient_data[i] = 2*((Patient_data[i] - min(Patient_data[i])) / ( max(Patient_data[i]) - min(Patient_data[i]) ))-1

print(Patient_data.to_string())

Patient_data_Y = Patient_data.iloc[:,10:]
Patient_data_X = Patient_data.iloc[:,:-1]

# Split dataset into training set and test set
Patient_data_X_train, Patient_data_X_test, Patient_data_Y_train, Patient_data_Y_test = train_test_split(Patient_data_X, Patient_data_Y, test_size=0.2,random_state=210) # 80% training and 20% test

classifier = GaussianNB()
classifier.fit(Patient_data_X_train, Patient_data_Y_train)

y_pred = classifier.predict(Patient_data_X_test)

ac = accuracy_score(Patient_data_Y_test ,y_pred)
cm = confusion_matrix(Patient_data_Y_test , y_pred)

print (ac)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity)

geometric_mean = (sensitivity*specificity)**(1/2)
print ('The Geometric Mean is: ' + str(geometric_mean))

print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.show()

cv = KFold(n_splits=5, random_state=1, shuffle=True)

# evaluate model
scores = cross_val_score(classifier, Patient_data_X, Patient_data_Y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print(scores)



#-------------------------------------------------------

Patient_data_X_rd = Patient_data_X[[0,1,4]]
#Patient_data_X_rd = Patient_data_X[[0,1]]
Patient_data_X_train_rd, Patient_data_X_test_rd, Patient_data_Y_train_rd, Patient_data_Y_test_rd = train_test_split(Patient_data_X_rd, Patient_data_Y, test_size=0.2,random_state=218)

classifier_reduced_dependency =  GaussianNB()
classifier_reduced_dependency.fit(Patient_data_X_train_rd, Patient_data_Y_train_rd)

y_pred_rd = classifier_reduced_dependency.predict(Patient_data_X_test_rd)

ac_rd = accuracy_score(Patient_data_Y_test_rd ,y_pred_rd)
cm_rd = confusion_matrix(Patient_data_Y_test_rd, y_pred_rd)

print (ac_rd)
print(cm_rd)

sensitivity_rd = cm_rd[0,0]/(cm_rd[0,0]+cm_rd[0,1])
print('Sensitivity : ', sensitivity_rd )

specificity_rd = cm_rd[1,1]/(cm_rd[1,0]+cm_rd[1,1])
print('Specificity : ', specificity_rd)

geometric_mean_rd = (sensitivity_rd*specificity_rd)**(1/2)
print ('The Geometric Mean is: ' + str(geometric_mean_rd))

disp_rd = ConfusionMatrixDisplay(confusion_matrix=cm_rd, display_labels=classifier_reduced_dependency.classes_)
disp_rd.plot()
plt.show()


cv_rd = KFold(n_splits=5, random_state=1, shuffle=True)

# evaluate model
scores_rd = cross_val_score(classifier_reduced_dependency, Patient_data_X_rd, Patient_data_Y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print(scores_rd)