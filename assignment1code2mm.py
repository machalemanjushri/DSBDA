#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


# In[6]:


iris = pd.read_csv(csv_url, header = None)


# In[7]:


col_names =['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']


# In[8]:


iris = pd.read_csv(csv_url, names = col_names)


# In[9]:


df1=df=iris


# In[10]:


iris.head(8)


# In[11]:


iris.tail()


# In[12]:


iris.index


# In[13]:


iris.columns


# In[14]:


iris.shape


# In[17]:


iris.dtypes


# In[19]:


iris.describe()


# In[20]:


iris.columns.values


# In[21]:


iris.iloc[5]


# In[22]:


iris[47:51]


# In[23]:


iris.loc[:,["Sepal_Length","Sepal_Width"]]


# In[24]:


cols_2_4=iris.columns[2:4]
iris[cols_2_4]


# In[25]:


iris.isnull().any()


# In[26]:


iris.isnull().sum()


# In[27]:


iris.dtypes


# In[28]:


df=iris
df['petal Length(cm)']=iris['Petal_Length'].astype("int")


# In[29]:


df1=df


# In[36]:


df


# In[37]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


# In[38]:


X=iris.iloc[:,:4]


# In[39]:


X


# In[40]:


X_scaled = min_max_scaler.fit_transform(X)


# In[41]:


df_normalized = pd.DataFrame(X_scaled)


# In[42]:


df_normalized


# In[43]:


df2=df
df2['Species'].unique()


# In[44]:


from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()


# In[45]:


features_df=df2.drop(columns=['Species'])


# In[46]:


features_df


# In[47]:


enc_df=(enc.fit_transform(df2[['Species']])).toarray()


# In[48]:


enc_df = pd.DataFrame(enc_df, columns = ['Iris-Setosa','Iris-Versicolor','Iris-Virginica'])


# In[49]:


df_encode = features_df.join(enc_df)


# In[50]:


df_encode


# In[ ]:




