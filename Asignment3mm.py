#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests


# In[4]:


df=pd.read_csv("C://Users//pcoec//Downloads//mall_customers - Sheet1(1).csv")


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[ ]:


import pandas as pd
import numpy as np
import requests
df=pd.read_csv("/content/Assign_3_Mall_Customers.csv")
df[['Age','Annual Income (k$)','Spending Score (1-100)']].mean()


# In[ ]:


df[['Age','Annual Income (k$)','Spending Score (1-100)']].median()


# In[ ]:


df[['Age','Annual Income (k$)','Spending Score (1-100)']].mean()


# In[ ]:


df[['Age','Annual Income (k$)','Spending Score (1-100)']].max()


# In[ ]:


df[['Age','Annual Income (k$)','Spending Score (1-100)']].std()


# In[ ]:


df2=df.groupby('Genre')
df2


# In[ ]:


for Genre,Genre_f in df2:
    print(Genre)
    print(Genre_f)


# In[ ]:


df2.get_group('Female')


# In[ ]:


df2.get_group('Male')


# In[ ]:


df2[['Age','Annual Income (k$)','Spending Score (1-100)']].max()


# In[ ]:


df[['Age','Annual Income (k$)','Spending Score (1-100)']].min()


# In[ ]:


df[['Age','Annual Income (k$)','Spending Score (1-100)']].mean()


# In[ ]:


df[['Age','Annual Income (k$)','Spending Score (1-100)']].std()


# In[ ]:


url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
r=requests.get(url)


# In[ ]:


df3=pd.read_csv(url)
df3


# In[ ]:


df3.columns=("A","B","C","D","E")
df3


# In[ ]:


df4=df3.groupby("E")
df4


# In[ ]:


df4.get_group("Iris-setosa")


# In[ ]:


df4.get_group("Iris-virginica")


# In[ ]:


df4.mean()


# In[ ]:


df4.std()


# In[ ]:


df4.min()


# In[ ]:


df4. max()


# In[ ]:


a=np.percentile(df3['A'],50)


# In[ ]:


a

