#!/usr/bin/env python
# coding: utf-8

# In[25]:


#math/data libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

#ml libs
import keras
from keras import backend as K
from keras.models import Sequential
from keras. layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical


# In[26]:


trainFile = pd.read_csv('./Data/train.csv').drop(columns="datasetId")
testFile = pd.read_csv('./Data/test.csv').drop(columns="datasetId")


# In[27]:


#train
train_samples = trainFile.drop(columns='condition').to_numpy()
train_labels = trainFile['condition'].to_numpy()

#test
test_samples = testFile.drop(columns='condition').to_numpy()
test_labels = testFile['condition'].to_numpy()


# In[6]:


#normalizing features
scaler = MinMaxScaler(feature_range=(0,1))
train_samples = scaler.fit_transform(train_samples)
test_samples = scaler.fit_transform(test_samples)

#one-hot-encoding labels
one_hot_encoder = OneHotEncoder(categories='auto')
train_labels = one_hot_encoder.fit_transform(train_labels.reshape(-1, 1)).toarray()
test_labels = one_hot_encoder.fit_transform(test_labels.reshape(-1, 1)).toarray()


# In[11]:


#build the model
model = Sequential([
    Dense(34, input_shape=[34,], activation='relu'),
#     Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(2, activation='softmax')
])


# In[12]:


model.summary()


# In[13]:


model.compile(Adam(lr=.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[14]:


model.fit(train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=3, shuffle=True, verbose=2)


# In[15]:


model.save('model.h5')


# In[18]:


y_pred=model.predict(test_samples)
y_pred =(y_pred>0.5)


# In[21]:


y_pred


# In[22]:


test_labels


# In[17]:


test_samples


# In[23]:


one_hot_encoder


# In[28]:


test_labels


# In[ ]:


import numpy as np

import pandas as pd
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt


# In[ ]:


trainFile = pd.read_csv('./Data/train.csv').drop(columns="datasetId")
testFile = pd.read_csv('./Data/test.csv').drop(columns="datasetId")


# In[ ]:


#features
X_train = trainFile.drop(columns='condition')
y_train = trainFile['condition']
X_test = testFile.drop(columns='condition')
y_test = testFile['condition']


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))


# In[ ]:


i=0
knn.predict([X_test.iloc[i]])

