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
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer




#Code that is referenced in all the functions below.
food_json = pd.read_json("data/yummly.json")
all_ingredients = set()
for recipe in food_json['ingredients']:
    for food in recipe:
        all_ingredients.add(food.lower())
all_ingredients = sorted(list(all_ingredients))  
     


class color:
#Color utility class.  This will provide some basic command line formatting
#and allow for important terms to be displayed in bol on a linux/unix OS
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   RED = '\033[91m'
   END = '\033[0m'

               
               
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
    if os.path.isfile("pickles/naive_bayes.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test    
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients']),row['cuisine']))
    trainSet = featureSet[:2000]
    testSet = featureSet[2000:2400]
    classifier = nltk.NaiveBayesClassifier.train(trainSet)
    print("Train/test accuracy using first 2400 observations: "+
          str(nltk.classify.accuracy(classifier,testSet)) + "\n")
    #Accuracy .5825
    print("Training full dataset . . .")
    classifier = nltk.NaiveBayesClassifier.train(featureSet)
    save_classifier = open("pickles/naive_bayes.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return
    
 
    
#Function that will train a multiclass logistic regression classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training   
def train_logit():
    if os.path.isfile("pickles/logit.pickle"):
        print("Model has already been trained")
        return
    #Create featureset and perform basic train/test  
    featureSet = []
    for index,row in food_json.iterrows():
        featureSet.append((make_features(row['ingredients']),row['cuisine']))
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
    save_classifier = open("pickles/logit.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return 
 
    

#Function that will train a Multinomail Bayes classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training   
def train_multNB():
    if os.path.isfile("pickles/mult_nb.pickle"):
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
    save_classifier = open("pickles/mult_nb.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return 
   
    
    
    

#Function that will train stochastic gradient descent classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training
def train_sgd():
    if os.path.isfile("pickles/sgd.pickle"):
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
    #Accuracy was .62
    print("Training full dataset . . .")
    classifier = SGD_classifier.train(featureSet)
    save_classifier = open("pickles/sgd.pickle","wb")
    pickle.dump(classifier,save_classifier)
    save_classifier.close()
    return 

  
    
#Function that will train a Linear SVC classifier using the yummly.json
#file and save the trained classifier as a .pickle file.  If a .pickle file
#exists in the current working directory with the specified name, no classifer
#will be trained. Otherwise, a new .pickle file will be saved contain the
#classifier.  Note that the FULL data set is being used after training    
def train_linearSVC():
    if os.path.isfile("pickles/linear_SVC.pickle"):
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
    save_classifier = open("pickles/linear_SVC.pickle","wb")
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


#Function that finds and import the .pickle files and intializes an
#EnsembleClassifer class, which it returns          
def initialize_classifier():    
    pickles = ['pickles/linear_SVC.pickle',
               'pickles/logit.pickle',
               'pickles/mult_nb.pickle',
               'pickles/naive_bayes.pickle',
               'pickles/sgd.pickle']
    classifier_set = []
    for p in pickles:
        if os.path.isfile(p):
            classifier_imp=open(p,"rb")
            classifier_p = pickle.load(classifier_imp)
            classifier_imp.close()
            classifier_set.append(classifier_p)
        else:
            print("Missing " + str(p) + " - Please retrain\n")
            return
        
    ensemble_classifier = Ensemble_Classifier(classifier_set[0],classifier_set[1],
                                              classifier_set[2],classifier_set[3],classifier_set[4])
    return ensemble_classifier
    
   
    
    
#Compute the cosine similarity of the user-specified ingredients with
#all the recipes in the database.  The cosine similarity is computed based
#on the tf-idf vector,which is computed using the sklearn TfidfVectorizer 
#module.  Tfidf is chosen over a simple word count because Tfidf will
#prioritize recipes that have a similar number of ingredients as the supplied
#recipe.  If the user gives 4 ingredients for a recipe, the count vectorizer
#wouldn't discrimate between two recipes that match all 4 ingredients, but
#one recipe has 20 more ingredients that the other.  Tfidf, on the other hand,
#would select the recipe with fewer ingredients as more similar.  Through tfidf,
#we get a better sense of proportion
def compute_similarity(ingreds):
    #Make each ingredient list a single string
    foods = ['|'.join(x) for x in food_json['ingredients']] 
    #test example input from user:
    _recipe = '|'.join(ingreds)
    vect = TfidfVectorizer(tokenizer=lambda x: x.split('|'), 
                           vocabulary=all_ingredients)
    vect.fit(foods)
    X = vect.transform(foods)
    Y = vect.transform([_recipe])    
    cosine_similarities = linear_kernel(Y,X).flatten()
    
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]
    return related_docs_indices



#Function to maintain the proper order of function calls when option 3
#is selected from teh user menu.  Accepts a list of user specfied ingredients
#and makes a cuisine-type prediction, as well as calculates the 5 most similar
#ingredients.  Also outputs results to the command line with formatting.
def run_option_three(user_recipe):
    #Build and predict classifer
    prediction_classifier = initialize_classifier()
    test_recipe = make_features(user_recipe)
    
    one = ("Predicted type of cuisine for provided ingredients: "+color.BOLD + color.RED +
          str(prediction_classifier.classify(test_recipe)) + color.END)
    print(one)      
    two = ("With " + color.BOLD + color.RED +str(prediction_classifier.vote_percentage(test_recipe)) +
          "% majority vote (5 models)\n" + color.END)
    print(two)
    
        #Compute similarity scores with cosiine similarity
    print("The five most similar dishes are the following: ")
    related = compute_similarity(user_recipe)
    for i in related:
        print("ID: "+str(food_json.iloc[i]['id']))
    flag = False
    time.sleep(.1) #Added pause because printing was getting jumbled
    use_input = input("Would you like to see the ingredients for each of these dishes? (y/n): ")
    print() #formatting
    while not flag:
        if use_input.lower() == 'y':
            flag = True
            for j in range(len(related)):
                print("Dish " +str(j+1) + ": " + str(food_json.iloc[related[j]]['ingredients']) 
                + "\n")
        elif use_input.lower() == 'n':
            flag = True
        else:
            use_input = input("Invalid entry.  Please enter 'y' or 'n': ")
    return 
