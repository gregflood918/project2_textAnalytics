#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:58:26 2017

@author: gregflood918
"""

import os
import nltk
from nltk.classify import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from statistics import mode


import pickle
import pandas as pd

'''
import os, sys
lib_path = os.path.abspath(os.path.join('..','project1'))
sys.path.append(lib_path)
'''

'''

food_json = pd.read_json("yummly.json")
all_ingredients = set()
j = []
for recipe in food_json['ingredients']:
    for food in recipe:
        all_ingredients.add(food.lower())
all_ingredients = sorted(list(all_ingredients))    
'''


#Function that accepts a recipe (as an array) and returns a feature
#dictionary.  Each word represents a key and the value will be true or
#false depending on the presence or absence of the ingredient in the
#passed recipe.  Length of returned dicitonary will be equivalent to
#the number of unique ingredients.
def make_features(doc,all_ingredients):
    doc = [j.lower() for j in doc] #consistent lower case
    ingredients = set(doc)
    
    features = {}
    for food in all_ingredients:
        features[food] = (food in ingredients) #True if recipe has food
    return features    

#Linear kernel is often good for text:
#https://www.svm-tutorial.com/2014/10/svm-linear-kernel-good-text-classification/   
#So classifiers will be predominately 

   
#Function that will train a Naive Bayes classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training
def train_bayes(food_json, all_ingredients):
    if os.path.isfile("naive_bayes.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test    
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients'], all_ingredients),row['cuisine']))
    trainSet = featureSet[:2000]
    testSet = featureSet[2000:2400]
    classifier = nltk.NaiveBayesClassifier.train(trainSet)
    print("Train/test accuracy using first 1200 observations: "+
          str(nltk.classify.accuracy(classifier,testSet)) + "\n")
    #Accuracy .5825
    print("Training full dataset . . .")
    classifier = nltk.NaiveBayesClassifier.train(featureSet)
    save_classifier = open("naive_bayes.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return
    
 
    
#Function that will train a multiclass logistic regression classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training   
def train_logit(food_json, all_ingredients):
    if os.path.isfile("logit.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test  
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients'], all_ingredients),row['cuisine']))
    #Create featureset and perform basic train/test    
    trainSet = featureSet[:2000]
    testSet = featureSet[2000:2400]
    Logreg_classifier = SklearnClassifier(LogisticRegression())
    classifier = Logreg_classifier.train(trainSet)
    print("Train/test accuracy using first 2400 observations: "+
          str(nltk.classify.accuracy(classifier,testSet)) + "\n")
    #.655 Accuracy
    print("Training full dataset . . .")
    classifier = Logreg_classifier.train(featureSet)
    save_classifier = open("logit.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return 
 
    

#Function that will train a Multinomail Bayes classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training   
def train_multNB(food_json, all_ingredients):
    if os.path.isfile("mult_nb.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test  
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients'], all_ingredients),row['cuisine']))
    #Create featureset and perform basic train/test    
    trainSet = featureSet[:2000]
    testSet = featureSet[2000:2400]
    MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
    classifier = MultinomialNB_classifier.train(trainSet)
    print("Train/test accuracy using first 2400 observations: "+
          str(nltk.classify.accuracy(MultinomialNB_classifier,testSet)) + "\n")
    #Accuracy was .5575
    print("Training full dataset . . .")
    classifier = MultinomialNB_classifier.train(featureSet)
    save_classifier = open("mult_nb.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return 
   
    
    
    

#Function that will train stochastic gradient descent classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training
def train_sgd(food_json, all_ingredients):
    if os.path.isfile("sgd.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test  
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients'], all_ingredients),row['cuisine']))
    #Create featureset and perform basic train/test    
    trainSet = featureSet[:2000]
    testSet = featureSet[2000:2400]
    SGD_classifier = SklearnClassifier(SGDClassifier())
    classifier = SGD_classifier.train(trainSet)
    print("Train/test accuracy using first 2400 observations: "+
          str(nltk.classify.accuracy(SGD_classifier,testSet)) + "\n")
    #Accuracy was .62
    print("Training full dataset . . .")
    classifier = SGD_classifier.train(featureSet)
    save_classifier = open("sgd.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return 

  
    
#Function that will train a Linear SVC classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training    
def train_linearSVC(food_json, all_ingredients):
    if os.path.isfile("linear_SVC.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test  
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients'], all_ingredients),row['cuisine']))
    #Create featureset and perform basic train/test    
    trainSet = featureSet[:2000]
    testSet = featureSet[2000:2400]
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    classifier = LinearSVC_classifier.train(trainSet)
    print("Train/test accuracy using first 2400 observations: "+
          str(nltk.classify.accuracy(LinearSVC_classifier,testSet)) + "\n")
    #Accuracy was .6375
    print("Training full dataset . . .")
    classifier = LinearSVC_classifier.train(featureSet)
    save_classifier = open("linear_SVC.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return 

    
#Define an ensemble classifier class that will accept all five of the
#previously trained models and make a prediction for the class of the user
#input ingredients based on a majority vote of the five models.  The models
#will be imported form .pickle files in order to save time.
class Ensemble_Classifier(ClassifierI):
        def __init__(self, *classifiers):
            self._classifiers = classifiers
        
        def classify(self, feature_vector):
            guess = []
            for i in self._classifiers:
                v = i.classify(feature_vector)
                guess.append(v)
            return mode(guess) #Returns class that has a majority
            #Not sure about the tie breaker here...
        def vote_percentage(self, feature_vector):
            guess = []
            for i in self._classifiers:
                v = i.classify(feature_vector)
                guess.append(v)
            return guess.count(mode(guess))/ len(guess) * 100



           
def initialize_classifier():
    
    pickles = ['pickles/linear_SVC.pickle',
               'pickles/logit.pickle',
               'pickles/mult_nb.pickle',
               'pickles/naive_bayes.pickle',
               'pickles/sgd.pickle']
    classifier_set = []
    for p in pickles:
        if os.path.isfile(p):
            classifier_imp=open(pickles,"rb")
            classifier_p = pickle.load(classifier_imp)
            classifier_imp.close()
            classifier_set.append(classifier_p)
        else:
            print("Missing " + str(p) + " - Please retrain\n")
            return
        
    ensemble_classifier = Ensemble_Classifier(classifier_set[0],classifier_set[1],
                                              classifier_set[2],classifier_set[3],classifier_set[4])
    return ensemble_classifier
    
    
    
def make_prediction(food_json, all_ingredients):
    prediction_classifier = initialize_classifier()
    print("Predicted type of cuisine for provided ingredients: " +
          str(prediction_classifier.classify(make_features(['herring','butter','chive','rice'], all_ingredients))))
    print("With " + str(prediction_classifier.vote_percentage(make_features(['herring','butter','chive','rice'],
                                                                            all_ingredients))) + "% confidence")
    return 



#featureSet = []
#for index,row in food_json.iterrows():
#    featureSet.append((make_features(row['ingredients']),row['cuisine']))
        
#trainSet = featureSet[:4000]
#testSet = featureSet[4000:]
#classifier = nltk.NaiveBayesClassifier.train(trainSet)
#print(nltk.classify.accuracy(classifier,testSet))

#classifier_f=open("naive_bayes.pickle","rb")

#classifier = pickle.load(classifier_f)

#classifier.classify(make_features(['herring','butter','chive','rice']))



#clssifier_f.close()
