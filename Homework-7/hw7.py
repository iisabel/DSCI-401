#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:58:44 2023

@author: isabelheard
"""
  
import pandas as pd    
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.optimizers import Adam





#Download the pulsar dataset & Cleaning
train = pd.read_csv('/Users/isabelheard/Downloads/DSCI 401/DSCI-401/HW7/PulsarData/pulsar_data_train.csv')
train.isnull().sum()
train = train.fillna(train.mean())
train.isnull().sum()

test = pd.read_csv('/Users/isabelheard/Downloads/DSCI 401/DSCI-401/HW7/PulsarData/pulsar_data_test.csv')
test.isnull().sum()
#test = test.fillna(test.mean())

#Test & train data sets
X = train.drop('target_class',axis=1)
y = train[['target_class']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)

#Scale
from sklearn.preprocessing import StandardScaler
ss=StandardScaler() 
X_trains = ss.fit_transform(X_train)
X_tests = ss.transform(X_test)


#Keras
model = Sequential()
model.add(Dense(16,activation='relu',input_dim=8))
model.add(Dropout(0.25))
model.add(Dense(8,activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)  
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

weight = {0 : 1., 1 : 2.}

history = model.fit(X_trains, y_train, epochs=100, batch_size=64, class_weight=weight, validation_data = (X_tests,y_test), verbose=2)


#plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()





#Make predictions on the test data
ps_test1 = test.copy()
X = ps_test1.drop('target_class', axis=1)
y = ps_test1[['target_class']]

imputer = SimpleImputer(strategy='mean')
ps_test1 = imputer.fit_transform(X)
ps_test1 =   pd.DataFrame(data=ps_test1,columns=X.columns)
ps_test1.isnull().sum()
ps_test1_scaled = ss.transform(ps_test1)

#Test on Keras
predictions = model.predict(ps_test1_scaled)
rounded_predictions = [round(x[0]) for x in predictions]



#Fill the 'target_class' column in the test dataset with the predicted values
test['target_class'] = rounded_predictions
result_df = pd.DataFrame(test)

#Save the DataFrame to a CSV file
result_df.to_csv('/Users/isabelheard/Downloads/DSCI 401/DSCI-401/HW7/predictions2.csv', index=False)







#Plot
result_df.head()
import matplotlib.pyplot as plt
class_counts = result_df['target_class'].value_counts()
labels = ['Class 0', 'Class 1']
plt.pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
plt.title('Distribution of Target Classes')
plt.show()

