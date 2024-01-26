#!/usr/bin/env python
# coding: utf-8

# **Importing Dependencies**

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# **Data Collection and Data Processing**

# In[2]:


# Loading the dataset to a pandas Dataflow
sonar_data = pd.read_csv('sonar.csv', header= None)


# In[3]:


sonar_data.head()


# In[4]:


# Number of Rows and Columns
sonar_data.shape


# In[5]:


sonar_data.describe()  #Describe  --> statistical measures of the data


# In[6]:


sonar_data[60].value_counts()


# M --> Mine
# 
# R --> Rock

# In[7]:


sonar_data.groupby(60).mean()


# In[8]:


# Seperating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]


# In[10]:


print(X)
print(Y)


# **Training and Test Data**

# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)


# In[12]:


print(X.shape, X_train.shape, X_test.shape)


# **Model Training**

# In[13]:


model = LogisticRegression()


# In[14]:


# Training the Logistic Regression model with data
model.fit(X_train,Y_train)


# **Model Evaluation**

# In[15]:


#Accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy for training data : ",training_data_accuracy*100,"%")


# In[16]:


#Accuracy on the Test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy for training data : ",test_data_accuracy*100,"%")


# **Making a Predictive System**

# In[17]:


input_data = (0.0100,0.0171,0.0623,0.0205,0.0205,0.0368,0.1098,0.1276,0.0598,0.1264,0.0881,0.1992,0.0184,0.2261,0.1729,0.2131,0.0693,0.2281,0.4060,0.3973,0.2741,0.3690,0.5556,0.4846,0.3140,0.5334,0.5256,0.2520,0.2090,0.3559,0.6260,0.7340,0.6120,0.3497,0.3953,0.3012,0.5408,0.8814,0.9857,0.9167,0.6121,0.5006,0.3210,0.3202,0.4295,0.3654,0.2655,0.1576,0.0681,0.0294,0.0241,0.0121,0.0036,0.0150,0.0085,0.0073,0.0050,0.0044,0.0040,0.0117)

##Changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
    print("The object is a Rock")
else:
    print("The object is a Mine")


# In[ ]:




