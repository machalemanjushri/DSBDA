#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
#matplotlib.style.use("ggplot")
import seaborn as sns


# In[ ]:


from google.colab import files
uploaded=files.upload()


# In[ ]:


import io
df=pd.read_csv(io.BytesIO(uploaded['IRIS[1].csv']))


# In[ ]:


df.info


# In[ ]:


print(df.shape)


# In[ ]:


print(df.head(20))


# In[ ]:


df.dtypes


# In[ ]:


print(df.groupby('species').size())


# In[ ]:


df.species.value_counts()


# In[ ]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[ ]:


df['species'].unique()


# In[ ]:


df['species']=label_encoder.fit_transform(df['species'])


# In[ ]:


df['species']

