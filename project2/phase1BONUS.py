#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:48:04 2017

@author: gregflood918
"""


import nltk
import csv
import re
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster


#make distance matrix from the srep00196-s2.csv file
#each ingredient will be a row, and each column will be also
#be ingredients.  The intersection will represent the distance between
#ingredients as represented by the number of chemical compounds they
#share.  To get distance, we calculate:
#  
#distance(i,j) = 1 / (similarity(i,j)+1)
#
#where the similarity of compunds i and j is the number of shared compounds
#
#So going row by row, we can calculate



#VERY USEFUL, PRETTY CODE FROM https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata



df = pd.read_csv("srep00196-s2.csv",header=4,sep=',',names=['a','b','sim'])



paired_similarity=[]
unique_ingredients = set()
with open('srep00196-s2.csv',newline='') as csvfile:
    filereader = csv.reader(csvfile, delimiter = ',')
    for i in range(200000): #We can limit the file size here
        if(i < 4):
            next(filereader)#skip header in file
        else:
            temp = next(filereader)
            #Replace _ with space characters
            for j in range(len(temp)):
                temp[j]=temp[j].replace("_"," ")
                if j==0 or j==1:
                    unique_ingredients.add(temp[j]) #To get length of dist matrix
            paired_similarity.append(temp)
#Initialize numpy similarity matrix
similarity = np.ones((len(unique_ingredients),len(unique_ingredients)))
#Cast unique ingredient set to a list
unique_ingredients = list(unique_ingredients)
#Make pandas dataframe
distance = pd.DataFrame(similarity,index=unique_ingredients,columns = unique_ingredients)
#populate similarity DF
for item in paired_similarity:
    distance[item[0]][item[1]] = (1/(1+int(item[2])))
#arbitrarily set the similarity of an item to itself to 100
for i in range(len(similarity)):
    d.iloc[i][i] = 0

#generate linkage matrix
Z = linkage(similarity.values,metric='cosine',method='complete')


dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=40,
    color_threshold=40
)
plt.show()



max_d = .03
clusters = fcluster(Z, max_d, criterion='distance')
clusters

zipped = list(zip(similarity.index.tolist(),clusters.tolist()))

print([x for x,y in zipped if y == 1])

print([x for x,y in zipped if y == 2])

print([x for x,y in zipped if y == 3])

print([x for x,y in zipped if y == 4])

print([x for x,y in zipped if y == 115])



'''
df.index.tolist() #index to a list
similarity.index.tolist()



#zip is a geneartor, list(zip()) list automatically exhausts an interator
to get a new list
[x for x,y in zipped if y == 140] #extract values with list comprehensions

[x for x,y in zipped if y == 1]
Out[159]: ['rum']

[x for x,y in zipped if y == 2]
Out[160]: ['rose wine']

[x for x,y in zipped if y == 3]
Out[161]: ['sherry']

[x for x,y in zipped if y == 4]
Out[162]: ['tomato']

[x for x,y in zipped if y == 5]
Out[163]: ['sauvignon grape']

#All of these are quite similar.  Why are they merging so very late?






list(zip(similarity.index.tolist(),clusters.tolist()))
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
'''
'''

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)


X
Out[96]: 
<9x33 sparse matrix of type '<class 'numpy.float64'>'
	with 47 stored elements in Compressed Sparse Row format>
 '''