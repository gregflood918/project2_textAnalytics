#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 18:22:02 2017

@author: gregflood918
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:08:27 2017

@author: gregflood918
"""
'''
scikit-learn already provides pairwise metrics (a.k.a. kernels in machine learning parlance) that work for both dense and sparse representations of vector collections. In this case we need a dot product that is also known as the linear kernel
'''

import os
import json
import nltk
import pickle
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
'''
for name, values in df.iteritems():
    print '{name}: {value}'.format(name=name, value=values[0])
    '''
    
''''
food_json = pd.read_json("yummly.json")
food_json = food_json[:5000] #for practice

#First step, create a list of ALL words!!!!
all_ingredients = set()
j = []
for recipe in food_json['ingredients']:
    for food in recipe:
        #x = re.sub(' ','_',food)  #REMOVE THIS LINE IF THINGS GET WEIRD
        all_ingredients.add(food.lower())

all_ingredients = sorted(list(all_ingredients)) 
'''   

#NEED TO CHANGE THIS SO THAT IT ACCEPTS AN ARRAY OF MEALS
#Make each ingredient list a single string
def compute_similar_meals(meal,f_json,unique_ingredients):
    foods = ['|'.join(x) for x in f_json['ingredients']]
        
    #test example input from user:
    vect = CountVectorizer(tokenizer=lambda x: x.split('|'), 
                           vocabulary=unique_ingredients)
    vect.fit(foods)
    X = vect.transform(foods)
    Y = vect.transform(meal)
   
    cosine_similarities = linear_kernel(Y,X).flatten()    
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    print("The 5 most similar meals: ")
    for i in range(len(related_docs_indices)):
        print(str(i+1) + "- ID: " + str(related_docs_indices[i]))
        print(f_json.iloc[i]['ingredients'])
        print()
    return
