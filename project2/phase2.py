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





food_json = pd.read_json("yummly.json")
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

#Linear kernel is often good for text:
#https://www.svm-tutorial.com/2014/10/svm-linear-kernel-good-text-classification/   
#So classifiers will be predominately 

   
#Function that will train a Naive Bayes classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training
def train_bayes():
    if os.path.isfile("naive_bayes.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test    
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients']),row['cuisine']))
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
def train_logit():
    if os.path.isfile("logit.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test  
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients']),row['cuisine']))
    #Create featureset and perform basic train/test    
    #trainSet = featureSet[:2000]
    #testSet = featureSet[2000:2400]
    Logreg_classifier = SklearnClassifier(LogisticRegression())
    #classifier = Logreg_classifier.train(trainSet)
    #print("Train/test accuracy using first 2400 observations: "+
    #      str(nltk.classify.accuracy(classifier,testSet)) + "\n")
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
def train_multNB():
    if os.path.isfile("mult_nb.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test  
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients']),row['cuisine']))
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
   
    
    
    

#Function that will train a Naive Bayes classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training
def train_sgd():
    if os.path.isfile("sgd.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test  
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients']),row['cuisine']))
    #Create featureset and perform basic train/test    
    trainSet = featureSet[:2000]
    testSet = featureSet[2000:2400]
    SGD_classifier = SklearnClassifier(SGDClassifier())
    classifier = SGD_classifier.train(trainSet)
    print("Train/test accuracy using first 2400 observations: "+
          str(nltk.classify.accuracy(SGD_classifier,testSet)) + "\n")
    #Accuracy was .6375
    print("Training full dataset . . .")
    classifier = SGD_classifier.train(featureSet)
    save_classifier = open("sdg","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return 

  
    
#Function that will train a Naive Bayes classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training    
def train_linearSVC():
    if os.path.isfile("linear_SVC.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test  
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients']),row['cuisine']))
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

    

class Ensemble_Classifier(ClassifierI):
        def __init__(self, *classifiers):
            self._classifier = classifiers
        
        def classify(self, feature_vector):
            guess = []
            for i in self._classifiers:
                v = i.classify(feature_vector)
                guess.append(v)
            return mode(guess) #Returns class that has a majority
            #Not sure about the tie breaker here...


    
    

''' 
featureSet = []
for index,row in food_json.iterrows():
    featureSet.append((make_features(row['ingredients']),row['cuisine']))
        
trainSet = featureSet[:4000]
testSet = featureSet[4000:]
classifier = nltk.NaiveBayesClassifier.train(trainSet)
print(nltk.classify.accuracy(classifier,testSet))

classifier_f=open("naive_bayes.pickle","rb")

classifier = pickle.load(classifier_f)

classifier.classify(make_features(['herring','butter','chive','rice']))



clssifier_f.close()


              '''