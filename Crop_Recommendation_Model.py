
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r'C:\Users\waran\Downloads\crop\Harvestify-master\Harvestify-master\Data-processed\crop_recommendation.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.size


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df['label'].unique()


# In[9]:


df.dtypes


# In[10]:


df['label'].value_counts()


# In[11]:


#sns.heatmap(df.corr(),annot=True)


# ### Seperating features and target label

# In[14]:


features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']


# In[15]:


# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []


# In[16]:


# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# # Decision Tree

# In[17]:


# In[18]:


from sklearn.model_selection import cross_val_score




# # Random Forest

# In[32]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[33]:


# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
score


# ### Saving trained Random Forest model

# In[34]:

'''
import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = r'D:\crop1\Harvestify-master\app\croprec\new folder\RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()
'''

import pickle
pickle.dump(RF, open('RandomForest.pkl', 'wb'))


# ## Accuracy Comparison

# In[41]:


plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# In[42]:


accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)


# ## Making a prediction

# In[36]:


data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)


# In[37]:


data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction = RF.predict(data)
print(prediction)


# In[ ]:



