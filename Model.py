#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  
print(np.__version__)

import pandas as pd
print(pd.__version__) 

import re  
print(re.__version__)

import nltk  
print(nltk.__version__) 

from sklearn.datasets import load_files  


nltk.download('stopwords')  


import pickle  
print(pickle.format_version)

from nltk.corpus import stopwords


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data = pd.read_csv('latest_ticket_data.csv')
data.head(5)


# In[10]:


data['target'] = data.Category.astype('category').cat.codes


# In[11]:


data.drop('Category', axis = 1, inplace=True)


# In[12]:


data.head()


# In[13]:


x = data['Description']
y = data['target']
from sklearn.model_selection import train_test_split  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[14]:


#type(data['Description'][1])
print(type(y))


# In[ ]:





# In[ ]:





# In[15]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1,2),stop_words=stopwords.words('english'))  
X_1 = vectorizer.fit_transform(x_train)


# In[16]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidfconverter = TfidfTransformer()

X_tfidf = tfidfconverter.fit_transform(X_1)


# In[17]:


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_tfidf, y_train)


# In[18]:


from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),stop_words=stopwords.words('english'))), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])

text_clf.fit(x, y)


# In[19]:


from sklearn.metrics import accuracy_score

pred = text_clf.predict(x_test)
acc = accuracy_score(y_test, pred)
acc


# In[20]:


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)


# In[22]:


from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
import itertools

encoder = LabelEncoder()
encoder.fit(['Application', 'Database', 'Network', 'Security', 'User Maintenance'])
text_labels = encoder.classes_ 
cnf_matrix = confusion_matrix(y_test, pred)
plt.figure(figsize=(24,20))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
plt.show()


# In[24]:


from sklearn.externals import joblib

joblib.dump(text_clf, 'text_classifier.pkl')


# In[28]:


from sklearn.externals import joblib

joblib.load('text_classifier.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




