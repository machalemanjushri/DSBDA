#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
titanic = sns.load_dataset("titanic")
titanic


# In[2]:


titanic.info()


# In[4]:


x=titanic["fare"]
x


# In[5]:


titanic.describe()


# In[6]:


titanic.info()


# In[7]:


titanic_cleaned=titanic.drop(['pclass','embarked','deck','embark_town'],axis=1)
titanic_cleaned.head(15)


# In[8]:


titanic_cleaned.info()


# In[9]:


titanic_cleaned.isnull().sum()


# In[11]:


titanic_cleaned.corr(method='pearson')


# In[12]:


sns.histplot(data=titanic,x="fare",bins=8)


# In[13]:


sns.histplot(data=titanic,x="fare",binwidth=10)


# In[14]:


sns.histplot(data=titanic,x="fare",bins=20,binwidth=10)


# In[15]:


sns.histplot(data=titanic,x="fare",binwidth=20)


# In[16]:


sns.histplot(data=titanic,x="fare",binwidth=1)


# In[17]:


sns.histplot(data=titanic,x="fare",bins=20,binwidth=50)


# In[ ]:




