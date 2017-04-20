#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:22:49 2017

@author: gregflood918
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:31:11 2017

@author: gregflood918
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.cm as cm

#Read file as a Pandas dataframe
paired_ingredients = pd.read_csv("srep00196-s2.csv", header=4, sep=',',
                                 names=['a', 'b', 'sim'])
#Replace underscores with spaces
paired_ingredients.replace('_',' ',regex=True,inplace=True)

#Extract list of unique ingredients
a = list(paired_ingredients['a'])
b = list(paired_ingredients['b'])
unique_ingredients = list(set(a + b))

#Build distance matrix : dist = 1/(1+sim)
dist = np.ones((len(unique_ingredients), len(unique_ingredients)))
dist = pd.DataFrame(dist, index=unique_ingredients, columns=unique_ingredients)
for index, row in paired_ingredients.iterrows():
    dist[row['a']][row['b']] = (1/(1+int(row['sim'])))
    dist[row['b']][row['a']] = (1/(1+int(row['sim'])))
for i in range(len(dist)):
    dist.iloc[i][i] = 0

#Compute similarity matrix
#Note that this is now scale to between 0 and 1
#Two of the same ingredient have a similarity of 1
sim = 1-dist

num_clusters = 8
#Perform Kmeans clustering
km = KMeans(n_clusters=num_clusters,random_state = 101)
km.fit(sim)
#Get Cluster labels for each item
clusters = km.labels_.tolist()

#Create useful dataframe containing ingredients and their corresponding clusters
foods = { 'ingredient': sim.index.tolist(), 'cluster': clusters}
frame = pd.DataFrame(foods, columns = ['ingredient','cluster'])
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
#Extract labels by taking the five most centrally located
#items from each group
labels = []
for i in range(num_clusters):
    innerLab=[]
    for ind in order_centroids[i, :]: #replace 6 with n words per cluster
        if clusters[ind]==i:
            innerLab.append(frame.iloc[ind]['ingredient'])
        if len(innerLab)==3:
            labels.append(innerLab)
            break
#Labels to strings
labels = [','.join(x) for x in labels]
        
#Perform PCA on the data
#Reduce dimensionality to 2 for visualization of clusters
pca = PCA(n_components=2) 
pca_space = pca.fit_transform(sim)
xs, ys = pca_space[:, 0], pca_space[:, 1] #Scale the pca vectors 
xs = scale(xs)
ys = scale(ys)
df2 = pd.DataFrame(dict(x=xs, y=ys, label=clusters))
groups = df2.groupby('label')

#cluster map
colors = cm.jet(np.linspace(0, 1, 8))
color_map = {}
label_map = {}

#Visualization - Show each ingredient as a point, with a color that corresponds
#to the cluster
for i in range(num_clusters):
    color_map[i] = colors[i]
    label_map[i] = labels[i]

for name,group in groups:
    plt.scatter(group.x,group.y,color=color_map[name],label=label_map[name])

plt.legend(numpoints=1,fontsize='x-small', loc=2, borderaxespad=0)
plt.title('PCA Representation of K-means')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.ylim((-4,7.5))
plt.show()
