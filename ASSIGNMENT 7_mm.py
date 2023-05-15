#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')


# In[2]:


from nltk import word_tokenize, sent_tokenize
sent = "Sachin is considered to be one of the greatest cricket players. Virat is the captain of the Indian cricket team"
print(word_tokenize(sent))
print(sent_tokenize(sent))


# In[3]:


from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
print(stop_words)


# In[4]:


token = word_tokenize(sent)
cleaned_token = []
for word in token:
 if word not in stop_words:
    cleaned_token.append(word)

print("This is the unclean version : ",token)
print("This is the cleaned version : ",cleaned_token)


# In[5]:


words = [cleaned_token.lower() for cleaned_token in cleaned_token if cleaned_token.isalpha()]


# In[6]:


print(words)


# In[7]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
port_stemmer_output = [stemmer.stem(words) for words in words]
print(port_stemmer_output)


# In[9]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
lemmatizer_output = [lemmatizer.lemmatize(words) for words in words]
print(lemmatizer_output)


# In[ ]:


from nltk import pos_tag
import nltk
nltk.download('averaged_perceptron_tagger')
token = word_tokenize(sent)
cleaned_token = []
for word in token:
 if word not in stop_words:
    cleaned_token.append(word)
tagged = pos_tag(cleaned_token)
print(tagged)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# In[ ]:


docs = [ "Sachin is considered to be one of the greatest cricket players",
 "Federer is considered one of the greatest tennis players",
 "Nadal is considered one of the greatest tennis players",
 "Virat is the captain of the Indian cricket team"]


# In[ ]:


vectorizer = TfidfVectorizer(analyzer = "word", norm = None , use_idf = True , smooth_idf=True)
Mat = vectorizer.fit(docs)
print(Mat.vocabulary_)


# In[ ]:


tfidfMat = vectorizer.fit_transform(docs)


# In[ ]:


print(tfidfMat)


# In[ ]:


features_names = vectorizer.get_feature_names_out()
print(features_names)


# In[ ]:


dense = tfidfMat.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist , columns = features_names)


# In[ ]:


df


# In[ ]:


features_names = sorted(vectorizer.get_feature_name())


# In[ ]:


docList = ['Doc 1','Doc 2','Doc 3','Doc 4']
skDocsIfIdfdf = pd.DataFrame(tfidfMat.todense(),index = sorted(docList), columns=features_names)
print(skDocsIfIdfdf)


# In[ ]:


csim = cosine_similarity(tfidfMat,tfidfMat)


# In[ ]:


csimDf = pd.DataFrame(csim,index=sorted(docList),columns=sorted(docList))


# In[ ]:


print(csimDf)

