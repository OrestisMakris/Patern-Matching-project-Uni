#Importing the necessary libraries 
#Numpy for the math Operations  
import numpy as np
#Pandas library for data preprocessing
import pandas as pd
#Matplotlib and seaborn for plotting the data
import matplotlib.pyplot as plt
import seaborn as sns
#skit_learn library from model_selection package importing train_test_split in order to split the arrays into random train and test subsets.
from sklearn.model_selection import train_test_split
#skit_learn library from naive_bayes package we use GaussianNB in order to implement the naive bayes classifier.
from sklearn.naive_bayes import GaussianNB
#skit_learn library from metrics package importing confusion_matrix, accuracy_score in order to evaluate the model and ConfusionMatrixDisplay for plotting the matrix.
from sklearn.metrics import confusion_matrix,accuracy_score , ConfusionMatrixDisplay
#skit_learn library from model_selection package importing cross_val_score in order to use the k-fold cross validation method to evaluate our model with multiple folds. 
from sklearn.model_selection import cross_val_score , KFold


#Importing Project data from csv using the Pandas dataframe
Patient_data = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv" , header=None )


#Using replace function from Pandas data frame to encode the genders with the numeric values 0 Male 1 Female  
Patient_data[1].replace('Male' ,0,inplace=True)
Patient_data[1].replace('Female' ,1,inplace=True)

#Using replace function from Pandas data frame to replace the Class encoding with 0=1 , 1=2   
Patient_data[10].replace(1 ,0,inplace=True)
Patient_data[10].replace(2 ,1,inplace=True)

#Displaying all the data from the Pandas dataframe using to_string() method
print(Patient_data.to_string())

#Using dropna method from Pandas dataframe to remove all the rows that contains NULL or NAN values.
Patient_data=Patient_data.dropna()

#Using value_counts method from Pandas dataframe to return an object with the counts of unique values of the array .
print(Patient_data[10].value_counts())


#Using hist method from Pandas dataframe to display a histogram a representation of the distribution of data 
#for the last column of the data array which contains the classes.
Patient_data[10].hist(color = "darkCyan")
plt.show()


#Using the seaborn and matplotlib libraries we plot 10 histograms one per column (features) of the data array.
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


#The log() numpy method a mathematical function that calculates natural logarithm of echa allmento 
#of the array here for the first feature
Patient_data[0] = np.log(Patient_data[0])

#The log() numpy method a mathematical function that calculates natural logarithm of each element of the array 
#for the remaining features expect the gender.
for i in range(2,10):
   Patient_data[i] = np.log(Patient_data[i])


#Using the seaborn and matplotlib libraries we plot 10 histograms one per column (features)
#of the data array in order to see the result of the logarithmic process of the data.
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

#using seaborn heatmap method in order to plot the heatmap of the features in order to see the dependency of each feature with the another.
corr = Patient_data.iloc[:,:-1].corr(method='pearson')
cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
sns.heatmap(corr , vmax =1 , vmin =.3 , cmap=cmap ,  annot=True,square =True , linewidths = .2)
plt.show()


#Normalise the data of each feature between -1 and 1 expect the gender (column with index 1)
Patient_data[0] = 2*((Patient_data[0] - min(Patient_data[0])) / ( max(Patient_data[0]) - min(Patient_data[0]) ))-1
for i in range(2,10):
    Patient_data[i] = 2*((Patient_data[i] - min(Patient_data[i])) / ( max(Patient_data[i]) - min(Patient_data[i]) ))-1


#separate the data in two pandas data frames Patient_data_Y contains the class data 
#and Patient_data_X Y contains the features data
Patient_data_Y = Patient_data.iloc[:,10:]
Patient_data_X = Patient_data.iloc[:,:-1]

# Split dataset into training set and test set
Patient_data_X_train, Patient_data_X_test, Patient_data_Y_train, Patient_data_Y_test = train_test_split(Patient_data_X, Patient_data_Y, test_size=0.2,random_state=210) # 80% training and 20% test

#Begin the process of implementing Naive Bayes Classifier Erotima 2. 

# Using GaussianNB() classifier from scikit-learn library 
#in order to implement the Gaussian Naïve Bayes algorithm for classification.
classifier = GaussianNB()

# The fit method from scikit-learn library trains the algorithm on the training data, after the model is initialized.
classifier.fit(Patient_data_X_train, Patient_data_Y_train)

# the Predict Method from scikit-learn library given a trained model, it predicts the label of a new set of data.
y_pred = classifier.predict(Patient_data_X_test)

# The accuracy_score method from scikit-learn is a function that computes subset accuracy: the set of labels predicted 
# for a sample must exactly match the corresponding set of labels in Patient_data_Y_test
ac = accuracy_score(Patient_data_Y_test ,y_pred)

# The Predict Method from scikit-learn library given a trained model, it predicts the label of a new set of data.
cm = confusion_matrix(Patient_data_Y_test , y_pred)

print (ac)

# Computes the sensitivity of the algorithm from the confusion matrix
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity )

# Computes the specificity of the algorithm from the confusion matrix
specificity = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity)

# Computes the geometric mean from the sensitivity and specificity of the algorithm
geometric_mean = (sensitivity*specificity)**(1/2)
print ('The Geometric Mean is: ' + str(geometric_mean))

print(cm)
#implementing the naïve bayes algorithm using the 5-fold cross validation technique Erotima 3 

#K-Folds cross-validator Pprovides train/test indices to split data in train/test sets. Split dataset into k consecutive folds 
cv = KFold(n_splits=5, random_state=1, shuffle=True)

#cross_val_score is a function in the scikit-learn package which trains and tests a model over multiple folds of your dataset
scores = cross_val_score(classifier, Patient_data_X, Patient_data_Y, scoring='accuracy', cv=cv, n_jobs=-1)

#Show accuracy for each fold performance
print(scores)


#||----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------||
#||----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------||
#||----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------||
#||----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------||


#After using the heatmap to determine the dependency between the feature of the data set we retrain the classifier 
#This new naive bayes classifier is called reduced_dependency (rd) and all new variables are called rd_previous_name 


#Creating a new training data set using the features (ccolumns) with the least correlation with the other features  
Patient_data_X_rd = Patient_data_X[[0,1,4]]

# Split dataset into training set and test set
Patient_data_X_train_rd, Patient_data_X_test_rd, Patient_data_Y_train_rd, Patient_data_Y_test_rd = train_test_split(Patient_data_X_rd, Patient_data_Y, test_size=0.2,random_state=218)

# Using GaussianNB() classifier from scikit-learn library 
#in order to implement the Gaussian Naïve Bayes algorithm for classification for reduced dependency data.
classifier_reduced_dependency =  GaussianNB()

#The fit method from scikit-learn library trains the algorithm on the training data, after the model is initialized.
classifier_reduced_dependency.fit(Patient_data_X_train_rd, Patient_data_Y_train_rd)

# the Predict Method from scikit-learn library given a trained model, it predicts the label of a new set of data.
y_pred_rd = classifier_reduced_dependency.predict(Patient_data_X_test_rd)

# The accuracy_score method from scikit-learn is a function that computes subset accuracy: the set of labels predicted 
# for a sample must exactly match the corresponding set of labels in Patient_data_Y_test
ac_rd = accuracy_score(Patient_data_Y_test_rd ,y_pred_rd)

# Computes the geometric mean from the sensitivity and specificity of the algorithm NB with reduced dpendancie data 
cm_rd = confusion_matrix(Patient_data_Y_test_rd, y_pred_rd)

print (ac_rd)
print(cm_rd)

# Computes the sensitivity of the algorithm from the confusion matrix
sensitivity_rd = cm_rd[0,0]/(cm_rd[0,0]+cm_rd[0,1])
print('Sensitivity : ', sensitivity_rd )

# Computes the Specificity of the algorithm from the confusion matrix
specificity_rd = cm_rd[1,1]/(cm_rd[1,0]+cm_rd[1,1])
print('Specificity : ', specificity_rd)

# Computes the geometric mean from the sensitivity and specificity of the algorithm
geometric_mean_rd = (sensitivity_rd*specificity_rd)**(1/2)
print ('The Geometric Mean is: ' + str(geometric_mean_rd))

# Ploting Confusion matrix 
disp_rd = ConfusionMatrixDisplay(confusion_matrix=cm_rd, display_labels=classifier_reduced_dependency.classes_)
disp_rd.plot()
plt.show()

#implementing the naïve bayes algorithm using the 5-fold cross validation technique Erotima 3 

#K-Folds cross-validator Pprovides train/test indices to split data in train/test sets. Split dataset into k consecutive folds 
cv_rd = KFold(n_splits=5, random_state=1, shuffle=True)

#cross_val_score is a function in the scikit-learn package which trains and tests a model over multiple folds of your dataset
scores_rd = cross_val_score(classifier_reduced_dependency, Patient_data_X_rd, Patient_data_Y, scoring='accuracy', cv=cv, n_jobs=-1)

#Show accuracy for each fold performance
print(scores_rd)