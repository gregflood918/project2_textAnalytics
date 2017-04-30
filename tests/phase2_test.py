#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 13:16:42 2017

@author: gregflood918
"""

import os, sys
import sqlite3
import pytest
from project2 import phase2


"""
Simple set of tests to check the presence of the appropriate
pickle files that are necessary for building the ensemble
classifier.  There are also 2 tests to ensure that the
essential functions of phase2.py work.  Namely prediction
of food classifications and the computation of the most
similar ingredients.
"""

#Test that the yummly file is present
def test_hasData():
    assert os.path.isfile("data/yummly.json")

#Test for naive bayes
def test_nb():
    assert os.path.isfile("pickles/naive_bayes.pickle")

#Test for multinomial naive-bayes
def test_multnb():
    assert os.path.isfile("pickles/mult_nb.pickle")

#Test for stochastic gradient descent
def test_sgd():
    assert os.path.isfile("pickles/sgd.pickle")

#Test for logistic regression
def test_logit():
    assert os.path.isfile("pickles/logit.pickle")

#Test for linear svc
def test_linSVC():
    assert os.path.isfile("pickles/linear_SVC.pickle")
    
#Test that we can make an accurate prediction with the 
#ensemble classifier.  This test initializes a classifier
#then predicts the classifcation of a known recipe.  The
#recipe is taken from yummmly.json and is of the 'greek'
#category.  Test checks that prediction matches category.
def test_ensemble():
    ensemble_classifier = phase2.initialize_classifier()
    ingreds = [
      "romaine lettuce",
      "black olives",
      "grape tomatoes",
      "garlic",
      "pepper",
      "purple onion",
      "seasoning",
      "garbanzo beans",
      "feta cheese crumbles"
    ]
    test_recipe = phase2.make_features(ingreds)
    result = ensemble_classifier.classify(test_recipe)
    assert result == 'greek'
    
#Test that the similarity computation works.  The list of
#ingredients passed is equal to the ingredients at index 0
#in yummly.json.  The similarity computation should thus 
#match index 0 as the most similar recipe    
def test_similarity():
    ingreds = [
      "romaine lettuce",
      "black olives",
      "grape tomatoes",
      "garlic",
      "pepper",
      "purple onion",
      "seasoning",
      "garbanzo beans",
      "feta cheese crumbles"
    ]
    indices = phase2.compute_similarity(ingreds)
    assert indices[0] == 0