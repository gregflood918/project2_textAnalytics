#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:58:26 2017

@author: gregflood918
"""

import json
import nltk
import pickle
import pandas as pd
import nltk

'''
for name, values in df.iteritems():
    print '{name}: {value}'.format(name=name, value=values[0])
    '''
food_json = pd.read_json("yummly.json")
food_json = food_json[:5000] #for practice

#First step, create a list of ALL words!!!!
all_ingredients = set()
j = []
for recipe in food_json['ingredients']:
    for food in recipe:
        all_ingredients.add(food.lower())

all_ingredients = sorted(list(all_ingredients))    


#Function that accepts a recipe (as an array) and returns a feature
#dictionary.  Each word represents a key and the value will be true or
#false depending on the presence or absence of the ingredient in the
#passed recipe.  Length of returned dicitonary will be equivalent to
#the number of unique ingredients.
def make_features(doc):
    doc = [j.lower() for j in doc] #consistent lower case
    ingredients = set(doc)
    
    features = {}
    for food in all_ingredients:
        features[food] = (food in ingredients) #True if recipe has food
    return features    


featureSet = []
for index,row in food_json.iterrows():
    featureSet.append((make_features(row['ingredients']),row['cuisine']))
        
trainSet = featureSet[:4000]
testSet = featureSet[4000:]
classifier = nltk.NaiveBayesClassifier.train(trainSet)
print(nltk.classify.accuracy(classifier,testSet))
              