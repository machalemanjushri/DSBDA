#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import load_dataset
#titanic dataset
data = pd.read_csv("C://Users//pcoec//Downloads//titanic_train.csv")
#tips dataset
tips = load_dataset("tips")


# In[2]:


data['Sex'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


# In[7]:


plt.hist(data['Age'], bins=5)
plt.show()


# In[8]:


sns.distplot(data['Age']) 
plt.show()


# In[9]:


sns.scatterplot(tips["total_bill"], tips["tip"])


# In[10]:


sns.scatterplot(tips["total_bill"], tips["tip"], hue=tips["sex"])
plt.show()


# In[11]:


sns.scatterplot(tips["total_bill"], tips["tip"], hue=tips["sex"], style=tips['smoker'])
plt.show()


# In[12]:


sns.barplot(data['Pclass'], data['Age'])
plt.show()


# In[13]:


sns.barplot(data['Pclass'], data['Fare'], hue = data["Sex"])
plt.show()


# In[14]:


sns.boxplot(data['Sex'], data["Age"])


# In[15]:


sns.boxplot(data['Sex'], data["Age"], data["Survived"])
plt.show()


# In[16]:


sns.distplot(data[data['Survived'] == 0]['Age'], hist=False, color="blue") 
sns.distplot(data[data['Survived'] == 1]['Age'], hist=False, color="orange")
plt.show()


# In[17]:


pd.crosstab(data['Pclass'], data['Survived'])


# In[18]:


sns.heatmap(pd.crosstab(data['Pclass'], data['Survived']))


# In[19]:


sns.clustermap(pd.crosstab(data['Parch'], data['Survived']))
plt.show()


# In[ ]:




