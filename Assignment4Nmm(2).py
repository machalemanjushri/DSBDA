#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


# In[6]:


data = pd.DataFrame(housing.data)


# In[7]:


data.columns = housing.feature_names
data.head()


# In[8]:


#adding target variable to dataframe
data['PRICE']= housing.target


# In[9]:


data


# In[10]:


data.isnull().sum()


# In[11]:


#finding out the correlation between the features
corr = data.corr()
corr.shape


# In[12]:


#plotting the heatmap of cireelation between features
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Blues')


# In[29]:


x =data.drop(['PRICE'], axis =1)
y = data['PRICE']


# In[14]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(x,y, test_size =0.2,random_state =0)


# In[15]:


import sklearn 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model=lm.fit(xtrain, ytrain)


# In[16]:


xtrain


# In[17]:


ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)


# In[33]:


testdata=[[0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.0900]]


# In[19]:


test_pred = lm.predict(testdata)
test_pred


# In[20]:


df1=pd.DataFrame(ytrain_pred,ytrain)
df2=pd.DataFrame(ytest_pred,ytest)
df1


# df2

# Model Evaluation

# In[22]:


from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(ytest,ytest_pred)
print('MSE on test data:',mse)
mse1=mean_squared_error(ytrain_pred,ytrain)
print('MSE on traning data:',mse1)


# In[24]:


r2=lm.score(xtest,ytest)
rmse=(np.sqrt(mean_squared_error(ytest,ytest_pred)))
print('r-squared:{}'.format(rmse))


# In[25]:


plt.scatter(ytrain,ytrain_pred,c='blue',marker='o',label='Training data')
plt.scatter(ytest,ytest_pred,c='lightgreen',marker='s',label='Test data')
plt.xlabel('True values')
plt.ylabel('predicted')
plt.title("True values vs predicted value")
plt.legend(loc='upper left')
plt.plot()
plt.show()


# In[34]:


testdata=[[0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.0900]]


# In[35]:


test_pred=lm.predict(testdata)
test_pred


# In[ ]:




