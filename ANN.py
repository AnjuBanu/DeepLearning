
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


#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing Keras libraries

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,ReLU,ELU
from keras.layers import Dropout


#Initialising a ANN
classifier = Sequential()

#Adding input layer and first hidden layers
classifier.add(Dense(units = 6, kernel_initializer = "he_normal", activation = "relu", input_dim = 11))

#Adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "he_normal", activation = "relu"))

#Adding third hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "he_normal", activation = "relu"))

#Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))

#Compiling the ANN
classifier.compile (optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

#Fitting the model
model_history =  classifier.fit(X_train, y_train, validation_split = 0.33,  batch_size = 10, nb_epoch= 100)



#Predicting the model
y_pred = classifier.predict (X_test)
y_pred = y_pred > 0.5

#Creation of confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
print (cm)

#checking the accuracy
from sklearn.metrics import accuracy_score
pred_accu = accuracy_score (y_pred, y_test)
print (pred_accu)

print (model_history.history.keys())
plt.plot (model_history.history["accuracy"])
plt.plot (model_history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train','test'], loc="upper left")
plt.show()


plt.plot (model_history.history["loss"])
plt.plot (model_history.history["val_loss"])
plt.title("Model Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train','test'], loc="upper left")
plt.show()










