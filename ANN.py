
# Creating of firs Artificial neural network model

#Importing the necessary files
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
churn_copy = pd.read_csv("Churn_Modelling.csv")
X = churn_copy.iloc[:,3:13]
y = churn_copy.iloc[:, 13]


# Creating dummy variables
geography = pd.get_dummies(churn_copy["Geography"], drop_first = True)
gender = pd.get_dummies(churn_copy["Gender"], drop_first = True)

# Concatinating the dummy variables 
X = pd.concat([X, geography, gender], axis=1)

# Dropping unnecessary columns
X =  X.drop(["Geography", "Gender"], axis= 1)

# splitting the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,  random_state=0)

S
#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing Keras libraries

import keras
from keras.models import Sequential
from keras.layers import Dense
from Keras.layers import Leaky