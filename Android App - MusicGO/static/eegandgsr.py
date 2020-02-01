#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from os import listdir

sum1=0

data = pd.read_csv(r"./Data/meditating.csv") 
col=[]
total_rows=len(data)
sum1=sum1+total_rows
for i in range(total_rows):
    col.append('nostress')
data['result'] = col
   
data2 = pd.DataFrame()

data2=data2.append(data)
#data2=data2.drop(columns=['Unnamed: 1'])


# In[2]:


data = pd.read_csv(r"./Data/thinking.csv") 
col=[]
total_rows=len(data)
sum1=sum1+total_rows
for i in range(total_rows):
    col.append('stress')
data['result'] = col


data2=data2.append(data)
#data2=data2.drop(columns=['Unnamed: 1'])


# In[3]:


data2


# In[4]:


from sklearn.utils import shuffle
data2 = shuffle(data2)
data2 = data2.sample(frac=1).reset_index(drop=True)
import sklearn.utils
data2 = sklearn.utils.shuffle(data2)
data2 = data2.reset_index(drop=True)


# In[5]:


X = data2
y = pd.DataFrame(data=data2, columns=['result'])


# In[6]:


#data2=data2.astype(float)
data2.dtypes


# In[7]:


# creating a dict file  
gender = {'nostress': 0,'stress': 1} 

y.result = [gender[item] for item in y.result] 
 


# In[8]:


del X['result']


# In[27]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X.shape


# In[28]:


X1=X[0:300000]
y1=y[0:300000]


# In[29]:


X2=X[300001:307200]
y2=y[300001:307200]


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42, stratify=y)


# In[30]:


import xgboost as xgb

dtrain = xgb.DMatrix(data=X1, label=y1)
dtest = xgb.DMatrix(data=X2)

params = {
    'max_depth': 6,
    'objective': 'multi:softmax',  # error evaluation for multiclass training
    'num_class': 2,
    'n_gpus': 0
}

bst = xgb.train(params, dtrain)

pred = bst.predict(dtest)


# In[31]:


pred


# In[33]:


y2


# In[34]:


from sklearn.metrics import classification_report

print(classification_report(y2, pred))


# In[36]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y2, pred)
cm


# In[37]:


from sklearn.metrics import accuracy_score

predictions = [round(value) for value in pred]
# evaluate predictions
accuracy = accuracy_score(y2, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[38]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X1, y1)
pred2=clf.predict(X2)


# In[39]:


from sklearn.metrics import classification_report

print(classification_report(y2, pred2))


# In[41]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y2, pred2)
cm


# In[42]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X1,y1)

#Predict the response for test dataset
y_pred = clf.predict(X2)


# In[43]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y2, y_pred)
cm


# In[44]:


from sklearn.metrics import classification_report

print(classification_report(y2, y_pred))


# In[ ]:




