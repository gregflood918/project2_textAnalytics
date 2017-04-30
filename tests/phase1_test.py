#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:49:47 2017

@author: gregflood918
"""

'''
Simple test class for the functions of phase1.py
Checks that data is present and performs some of the 
functions using an abbreviated dataset
'''


from project2 import phase1
import os
from sklearn.cluster import KMeans

def test_hasData():
    assert os.path.isfile("data/srep00196-s2.csv")
 
#Import paried ingredients for testing
ref = phase1.getPairedIngredients()

#Test that getPairedIngredients() has worked
#by comparing to a known entry in srep00196 dataset
def test_getPaired():
    assert ref.iloc[0]['sim'] == 5

#Test that distance matrix computes properly.  All diagonals 
#should have a distance of 0 in the distance matrix.  So assert
#that this is the case, and also test that an entry in dist has 
#the expected value.
def test_distance():
    dist = phase1.computeDistanceMatrix(ref[:20])
    assert dist.iloc[0][0] == 0 and dist.iloc[0][1] == 1
    
#Test to test that the dimension reduce function works 
#properly.  This function returns a grouped pandas dataframe
#Test checks that the dataframe is as expected.
def test_dimensionReduce():
    sim = 1-phase1.computeDistanceMatrix(ref[:20])
    km = KMeans(n_clusters=4,random_state = 101)
    km.fit(sim)
    #Get Cluster labels for each item
    clusters = km.labels_.tolist()
    reduced = phase1.dimensionReduce(sim,clusters)
    names = [name for (name,group) in reduced] #Pull out group names
    assert names[0] == 0 and len(names)==4