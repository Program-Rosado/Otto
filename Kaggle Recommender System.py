#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import get_dummies, concat, read_json
from matplotlib.pyplot import hist, show, boxplot, subplots
import numpy as np
from numpy import arange, argmax
from sklearn.cluster import KMeans
from numpy import arange, argmax
import pandas as pd
from pathlib import Path
import gc
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import sklearn.cluster as cluster
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#create train set from json kaggle dataset and add one extra features to it.
data_path = Path('/kaggle/input/otto-recommender-system/')
sample_size = 100
train_df = pd.DataFrame()
chunks = pd.read_json(data_path / 'train.jsonl', lines=True, chunksize=sample_size)

for chunk in chunks:
    event_dict = {'session': [], 'aid': [], 'ts': [], 'type': []}
    
    for session, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
        for event in events:
            event_dict['session'].append(session)
            event_dict['aid'].append(event['aid'])
            event_dict['ts'].append(event['ts'])
            event_dict['type'].append(event['type'])
    train_df = pd.DataFrame(event_dict)
    
    break
        
train_df = train_df.reset_index(drop=True)
train_df["minutes"] = train_df[["session", "ts"]].groupby("session").diff(-1)*(-1/1000/60)
train_df['type'].replace({'clicks': 0, 'carts': 1,'orders':2}, inplace=True)
train_df.isna().sum()
train_df.drop(train_df[train_df['minutes'].isnull()].index)
train_df


# In[ ]:



#create test set from json kaggle dataset and add one extra features to it.
test_df = pd.DataFrame()
chunks = pd.read_json(data_path / 'test.jsonl', lines=True, chunksize=sample_size)
for chunk in chunks:
    event_dict = {'session': [],'aid': [],'ts': [],'type': []}
    for session, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
        for event in events:
            event_dict['session'].append(session)
            event_dict['aid'].append(event['aid'])
            event_dict['ts'].append(event['ts'])
            event_dict['type'].append(event['type'])
    test_df = pd.DataFrame(event_dict)
    break   
    
test_df = test_df.reset_index(drop=True)
test_df["minutes"] = test_df[["session", "ts"]].groupby("session").diff(-1)*(-1/1000/60)
test_df['type'].replace({'clicks': 0, 'carts': 1,'orders':2}, inplace=True)
test_df


# In[ ]:


#KNN Method + add one extra features to the 
ii = IterativeImputer()
ii.fit(train_df)
train_df = ii.transform(train_df)
test_df = ii.transform(test_df)
num_clusters=train_df.shape[0]//20
kmeans = cluster.KMeans(n_clusters=num_clusters, init='random',
    max_iter=300, 
    tol=1e-04, random_state=42)
kmeans.fit(train_df)
predictions  = kmeans.predict(train_df)
kmeans.cluster_centers_
for i in range(num_clusters):
    f=newData[newData['Clusters']==i]['aid']
    f = f.iloc[:20].to_string(index=False)
    f=f.strip()
    f=f.replace('\n', ',')
    newData.loc[newData.Clusters==i, 'Clusters'] = f

