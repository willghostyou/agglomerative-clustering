#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

X = np.array([[0.4,0.53],
    [0.22,0.38],
    [0.35,0.32],
    [0.26,0.19],
    [0.08,0.41],
    [0.45,0.3],
    [0.38,0.31],
    [0.18,0.25],])


# In[3]:


import matplotlib.pyplot as plt

labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()


# In[4]:


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(X, 'single')

labelList = range(1, 9)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()


# In[5]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)


# In[6]:


print(cluster.labels_)


# In[ ]:




