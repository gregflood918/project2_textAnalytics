#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:05:58 2017

@author: gregflood918

Greg Flood
CS 5970 - Text Analytics

Project 2 - Phase 1
The goal of this project is to read in a list of paired
similarities for various types of food and perform 
clustering to explore similairites/differences between
the foods

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.cm as cm




#Function to get the paired ingredients file.  Returns
#a pandas dataframe containing the formatted ingredients
def getPairedIngredients():
    paired_ingredients = pd.read_csv("data/srep00196-s2.csv", 
                                     header=4, sep=',',
                                 names=['a', 'b', 'sim'])
    #Replace underscores with spaces
    paired_ingredients.replace('_',' ',regex=True,inplace=True)
    return paired_ingredients
    
    
    
#Function to comput the difference matrix for the a passed pandas
#dataframe.  The frame should contain columns names 'a' and 'b', along
#with a column 'sim' showing the similarity of the two ingredients in
#terms of shared components
def computeDistanceMatrix(paired_ingredients):
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
    return dist
    
        

#Function to reduce the dimensions of the dataset down from >1000 down 
#to two.  Returns a dataframe of values with columns xs, ys, and clusters.
#Each row correspondes to the same food in the similarity/distance matrix,
#and the xs, ys refer to the position of the ingreident along the first and
#second principal component axis.  Accepts a similarity matrix and a vector
#of numerical clsuter names.
def dimensionReduce(sim,clusters):
    pca = PCA(n_components=2) 
    pca_space = pca.fit_transform(sim)
    xs, ys = pca_space[:, 0], pca_space[:, 1]
    #Scale the pca vectors for better visualization
    xs = scale(xs)
    ys = scale(ys)
    df2 = pd.DataFrame(dict(x=xs, y=ys, label=clusters))
    groups = df2.groupby('label')   
    return groups
    
 
    
#Function that accepts the returned value from dimensionReduce.  It should
#be a dataframe of coordinates grouped by cluster number.  This is then
#plotted in a 2D plane using the principal component coordinates.  The
#labels are the 3 items from each cluster that are closest to the centroid
#for each of the 8 clusters.  We can see that the groups make coherent sense.
def createFigure(groups,labels):    
    colors = cm.jet(np.linspace(0, 1, len(labels)))  
    color_map = {}
    label_map = {}
    for i in range(len(labels)):
        color_map[i] = colors[i]
        label_map[i] = labels[i]   
    for name,group in groups:
        plt.scatter(group.x,group.y,color=color_map[name],label=label_map[name])
    #Formatting
    plt.legend(numpoints=1,fontsize='x-small', loc=2, borderaxespad=0)
    plt.title('PCA Representation of K-means')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.ylim((-4,7.5)) #Make room for legend
    plt.show()
    return
    

#Function that performs k-means clustering and visualizes the clusters.  It
#requires a distance matrix and an optional number of clusters.  The default
#is 8, and all the analysis in the readme assumes 8 clusters.  But you could
#do otherwise if you so choose.  This includes a call to the dimensionReduce()
#and createFigure() functions.  Also extracts the labels from the created
#clusters by selecting the 3 ingredients closest to the centroid of each
#cluster.  These are the "canonical labels"
def kMeansCluster(dist,num_clusters=8):
    sim = 1-dist
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
    reduced = dimensionReduce(sim,clusters) #Reduce dimensions and group
    createFigure(reduced,labels) #Show figure
    


#Simple function to be called in the main method.  This executes all of the
#above methods in the proper order.
def executePhaseOne():
    paired = getPairedIngredients()
    dist = computeDistanceMatrix(paired)
    kMeansCluster(dist)