
Project 2 - Phase Two
Greg Flood
gflood@ou.edu

#See requirements.txt for the packages required to run this program
#README FOR PHASE ONE IS APPENDED TO THE BOTTOM OF THIS FILE!

Run 'python3 main.py' from the project2_textAnalytics directory to
run the program.

##############################################################################
##############################################################################

Assignment:
Phase Two- The primary goal of phase two of project 2 is to create an
interface that accepts a list of food ingredients inputted by the
user then performs two tasks:

1. Predicts the type (style) of cuisine based on the ingredients
2. Returns the indices of the 'most similar' recpipes to the provided
ingredient list.

Both of these tasks will be discussed indivudally in the next section.

The 'yummly.json' dataset is used in this project, which gives a 
number of recipes, each with a unique ID, a cuisine style, and
a list of ingreidents in a json format.  In total, there are 
39774 distinct recipes with 6703 unique ingredients in the datset.

Since the user interface includes an option for visualizing the 
ingredients as clusters using the phase 1 methodology, the 
Project 2 phase 1 readme is appended to the bottom of this file.



##############################################################################

Discussion

Part One: Cuisine Prediction

In order to predict the style of dish given a list of ingredients,
the program reads in the full 'yummly.json' dataset and creates features
through a 'bag of words' approach, except that multi-word ingredients are
treated as single features (ex: green pepper and pepper are distinct features).
This isn't a problem because the 'yummly.json' file is highly structured and
each ingredient in a recipe is an individual element in the ingredient list.

A set of 6703 unique ingredients was created from 'yummly.json.'  The approach
to prediction was to create an ensemble of classifiers using different 
methodologies.  Using ensembles of models is a well-known approach to reduce
the bias of individual models by assigning classifications based on a majority
vote of several models.  For this program, the following 5 models were used:

Linear Support-Vector Classifier: 0.6375
Logistic Regression: 0.655
Multinomial Naive Bayes: 0.5575
Stochastic Gradient Descent: 0.62
Naive Bayes: 0.5825

The first 4 models used the scikit-learn implementation, whereas the last
model was from NLTK.  These particular models are known to perform well within 
the text classiifcation domain. Each of them were pretrained using the full
dataset, although the predictive performance of each model was noted using
only the first 2000 (of 39774) distinct recipes to predict the style of 
recipes 2001-2400.  The performance is listed above, next to the model.
The pretrained models are stored in the .pickle files and are accessed and
utilized for prediction upon the user entering a list of ingredients. Trianing
these models with the full data set took several hours, thus storing the
models as .pickle files save substantial time.

When the user provides a list of ingredients, the list is transformed to 
a feature vector, consisting of 6703 dimensions.  Each dimension corresponds
to an ingredient, with the binary value determined by the presence or absence
of the ingredient in the provided recipe.  This feature vector is then used
within the 'EnsembleClassifier' instance method 'classify(features).'  The
EnsembleClassifier class is initialized with a list of classifiers and 
calls the classify function for each classifier to predict the cuisine of 
the provided feature vector.  Then, taking the majority vote, the EnsembleClassifier
returns the predicted class for the given feature vector.  Additionally,
the EnsembleClassifier has a method called 'vote_percentage' that returns
then percentage of the 5 models that voted for the modal classification.


2. Most Similar Recipes

Additionally, the most similar recipes are computed for the provided list
of ingredients.  This is performed by transforming the dataset and provided list
into a  term frequency - inverse document frequency (tfidf) matrix/vector using 
the Scikit-learn TfIdfVectorizer module, and then computing the cosine similarity
between each recipe in the dataset and the provided recipe.  The 5 closest 
matches are returned to the user.

The TfIdfVectorizor.fit function transforms the yummly.json ingredient list
into 6703 dimension matrix with each row corresponding to an individual 
recipe.  Unlike the CountVectorizer module, which fills in the columns of
each row based on the number of occurrence of a word in a provided string of
text (giving higher weight to more frequently occurring words), the inverse-document
frequency portion (denominator) of tfidf normalizes based on how frequently the
word is used across all the documents.  In this problem context, the end result
is that it will cause recipes sharing uncommon ingredients to be treated
as more similar.  Additionally, this caused recipes that are similar in
the number of ingredients to be a closer match versus using the CountVectorizer
method.  The similarity is computed with cosine similarity, which is widely
used in text classification as it is better with sparse vectors, which is
exactly what we have in this particular problem.




##############################################################################
##############################################################################

Language:
Python 3

##############################################################################
##############################################################################

Testing:


##############################################################################
##############################################################################

Instructions:

To run the program, simply navigate to the project2_textAnalytics folder and
execute the following command:

python3 main.py

main.py contains a call to the searchMenu() function, which provides a command
line interface for the user.  

Note that main.py must execute in the "./project2_textAnalytics" directory.
Otherwise, it won't be able to find the approriate data files  Also note
that the functions of phase1.py and phase2.py are meant to be called from the 
project2_textAnalytics directory. They also will not work if called form any 
other directory but  "./project2_textAnalytics"


The file structure of project2_textAnalytics is below:

project2_textAnalytics/
        project2/
                phase1.py
                phase2.py
                __init__.py
        pickles
                linear_SVC.pickle
                logit.pickle
                mult_nb.pickle
                naive_bayes.pickle
                sgd.pickle
        README
        setup.py
        setup.cfg
        requirements.txt
        main.py
        data/
           yummly.json
		srep00196-s2.csv
        tests/


##############################################################################
##############################################################################
Functions and Classes:
Each of the functions from phase2.py will be discussed briefly below.  Also,
the functions from main.py will be discussed.  For descriptions of the
functions from phase1.py, please see the appended phase 1 readme appended
to the bottom of this file.



def make_features(doc):
Function that accepts a recipe (as an array) and returns a feature
dictionary.  Each word represents a key and the value will be true or
false depending on the presence or absence of the ingredient in the
passed recipe.  Length of returned dicitonary will be equivalent to
the number of unique ingredients in the yummly dataset.



class Ensemble_Classifier(ClassifierI):
    def __init__(self, *classifiers):
            self._classifiers = classifiers     
    def classify(self, feature_vector)
    def vote_percentage(self, feature_vector)

Define an ensemble classifier class that will accept a list of all five 
previously trained models and make a prediction for the class of the user
input ingredients based on a majority vote of the five models.  The models
will be imported form .pickle files in order to save time.  Additionally, 
the vote_percentage member method provides an estimated level of confidence
for the strength of the prediction by returning the percentage of models
that predicted the modal class. 


       
def initialize_classifier():  
Function that finds and import the .pickle files and intializes an
EnsembleClassifer class, which it returns    



def compute_similarity(ingreds):
Compute the cosine similarity of the user-specified ingredients (ingreds) with
all the recipes in the database.  The cosine similarity is computed based
on the tf-idf vector,which is computed using the sklearn TfidfVectorizer 
module.  Tfidf is chosen over a simple word count because Tfidf will
prioritize recipes that have a similar number of ingredients as the supplied
recipe.  If the user gives 4 ingredients for a recipe, the count vectorizer
wouldn't discrimate between two recipes that match all 4 ingredients, but
one recipe has 20 more ingredients that the other.  Tfidf, on the other hand,
would select the recipe with fewer ingredients as more similar.  Through tfidf,
we get a better sense of proportion



def run_option_three(user_recipe):
Function to maintain the proper order of function calls when option 3
is selected from teh user menu.  Accepts a list of user specfied ingredients
and makes a cuisine-type prediction, as well as calculates the 5 most similar
ingredients.  Also outputs results to the command line with formatting.



The following functions train the classifiers used in the ensemble 
clasifier using the yummly.jsonfile and saves the trained classifier as a 
.pickle file.  If a .pickle file exists in the current working directory 
with the specified name, no classifer will be trained. Otherwise, a new 
.pickle file will be saved and contain the classifier.  Note that the FULL 
data set is being used after training, however, training on the first 
2000 recipes is used to get a sense of predictive performance.

def train_bayes():
def train_logit():
def train_multNB():
def train_sgd():
def train_linearSVC():

These models are not used in the command line interface, but are only 
included to show how the training of the models was performed.
        


From main.py :
 
def searchMenu():
User interface for the translation command line tool.  The user is allowed to
select between 3 different options:

1 - Help
2 - Visualize data through clusters
3 - Predict cuisine type and calculate similar meals
0 - Quit

Each option corresponds to a number of function calls.
Program will run until the user manually enter '0'. If '3' is selected,
the user will enter ingredients one at a time until they enter '0'.  The
prediction is returned, along with the indices of the 5 most similar 
ingredients.  The user is then asked whether they want to see the ingredients
in these 5 recipes.

If '2' is selected, phase1.py is executed exactly as required by the 
submission for Project2 phase 1.  For details on these functions, see the
readme for phase one, which is appended below.



def main():
Calls the search menu



##############################################################################
##############################################################################        
References:

Data taken from:


Ahn, Y, et.al. "Flavor network and the principles of food pairing." 
Scientific Reports, 2011.

Jovanovik, M, et. al.  "Inferring Cuisine - Drug Interations Using the Linked 
Data  Approach." Scientific Reports, 2015.

For ensemble classifiers:
Sentdex nltk ensemble classifiers :
https://www.youtube.com/watch?v=vlTQLb_a564&t=304s

Clustering Reference:
http://brandonrose.org/clustering






##############################################################################
##############################################################################


##############################################################################
##############################################################################


PROJECT 1 - PHASE 1 README 
(INCLUDED FOR REFERENCE)


##############################################################################
##############################################################################


##############################################################################
##############################################################################







Assignment:
Phase One- 

This phase consists of 2 primary goals:
(a) - Use Python NLTK or scikit-learn to cluster the documents
(b) - Use graphing libraries to visualize the cluster.  Use knowledge of 
items inside the cluster to add canonical labels.

The dataset 'srep00196-s2.csv' is to be used in this assignment.  It
consists of 3 columns, the first 2 giving a food item/ingredient and the
third giving the number of compounds shared between the 2 ingredients.

The technical approach for each phase will be discussed next:

##############################################################################

Part(a):

K-means clustering through the sklearn Kmeans module was selected for implementation.
The primary reason for this selection was the size of the 'srep00196-s2.csv' dataset.
Because the dataset is so large, computing hierarchical clustering would become 
expensive, as it requires the calculating the distance between each point and every
other observation at each merge step. K-means avoids this by only calculating
the distance to each of the centroids, 

In order to use K-means clustering, the data had to be turned into features.  Since
'srep00196-s2.csv' essentially provides pairwise similarity scores (since the
number of shared components is basically similarity), a feature vector can
be created for each unique ingredient by counting the distance from every
other unique ingredient as a feature.  For example

	Distance[i][j] = 1 / (1 + Similarity[i][j]) = Distance[j][i] 

Where Similarity[i][j] is the number of compounds shared between 
ingredients i and j according to 'srep00196-s2.csv'

Since the dataset doesn't provide the number of compounds that makes up 
each ingredient, it is impossible to give an ingredient a similarity score to 
itself.  Creating a distance matrix allows us to circumvent this, because
we know that the distance between any ingredient and itself must be 0.
Thus, we can convert back to similarity if we choose:	

	1 - Distance[i][j] = Similarity(i,j)

This similarity (or distance) matrix can be used as an input for the 
sklearn Kmeans implementation.  8 clusters are the default option
for the number of clusters.  This number was selected through trial and error, 
and ultimately produced the most sensible groups.  The random state of the
algorithm is initialized to 101, so the result should be the same every simulation.
However, upon testing, it was noted that the labels of clusters changed between
days, so perhaps there is another element of randomness in the code.
Furthermore, the function kMeansCluster() accepts an optional argument
specifying the number of clusters, but this is not accessible via the 
executePhaseOne() function that runs the simulation in order.


##############################################################################

Part(b):

To aid visualization of the clusters in two dimension (note that the distance matrix has over
1000 features, equal to the number of unique ingredients), the Principal Component 
Analysis (PCA) module from sklearn was utilized.  The general idea behind PCA
is that it tries to transform the coordinate axes to new planes that maximize the
variance in the data.  The first two vectors from PCA are orthogonal axes that meet
this goal.  Since we now have only two axes, we can easily visualize a scatter plot
of our clusters, where each point represents an ingredient, the x-axis represents
the first Principal Component, and the y-axis represents the second Principal
Component.  Additionally, the cluster labels can be mapped to the color for each
point in the scatter plot, allowing easy visualization of the different clusters.

The labels for each cluster are determined but the 3 ingredients that are closest
to the centroid for each cluster in terms of Euclidean distance.  This would be
the 3 points "most" representative of the cluster.  The cluster labels are as follows:

0 - mustard, grapefruit juice, truffle
1 - ables sibirica, eucalyptus globulius oil, monarda punctata
2 - bantu beer, beer, black tea
3 - israeli orange, california orange, orange
4 - palm, petitgrain lemon, hyacinth oil
5 - tuberose, acacia, orange flower
6 - smoked fatty fish, tuna, keta salmon
7-  nutmeg, ashantl pepper, tobasco pepper.

We can see that these labels make some intuitive sense, as we see beverages,
sour fruits, fish, and peppers in their own distinct groups.


##############################################################################
##############################################################################

Language:
Python 3

##############################################################################
##############################################################################

Testing:
Since there is no user-facing interface for phase1, there are no tests modules. 
Also, the project description does not say anything about creating tests.


##############################################################################
##############################################################################

Instructions:

To run the program, simply navigate to the project2_textAnalytics folder and
execute the following command:

python3 main.py

main.py contains a call to the function executePhaseOne(), which executes
the relevant functions of phase 1 in the proper sequential order.

Note that main.py must execute in the "./project2_textAnalytics" directory.
Otherwise, it won't be able to find the 'srep00196-s2.csv' file.  Also note
that the functions of phase1.py are meant to be called from the same directory.
They also will not work if called form any other directory but 
"./project2_textanalytics"

project2_textAnalytics/
        project2/
                phase1.py
                __init__.py
        README
        setup.py
        setup.cfg
        requirements.txt
        main.py
        data/
		srep00196-s2.csv
        tests/


##############################################################################
##############################################################################

Functions:
Each of the functions from phase1.py will be discussed briefly below

def getPairedIngredients():
Function to get the paired ingredients file.  Returns a pandas dataframe containing the
 formatted ingredients

def computeDistanceMatrix(paired_ingredients):
Function to compute the difference matrix for the a passed pandas dataframe.  
The frame should contain columns names 'a' and 'b', along with a column 'sim' 
showing the similarity of the two ingredients in terms of shared components.


def dimensionReduce(sim,clusters):
Function to reduce the dimensions of the dataset down from >1000 down  to two. 
Returns a dataframe of values with columns xs, ys, and clusters. Each row 
corresponds to the same food in the similarity/distance matrix, 
and the xs, ys refer to the position of the ingredient along the first and
second principal component axis.  Accepts a similarity matrix and a vector
of numerical cluster names.


def createFigure(groups,labels):  
Function that accepts the returned value from dimensionReduce.  It should
be a dataframe of coordinates grouped by cluster number.  This is then
plotted in a 2D plane using the principal component coordinates.  The
labels are the 3 items from each cluster that are closest to the centroid
for each of the 8 clusters.  


def kMeansCluster(dist,num_clusters=8):
Function that performs k-means clustering and visualizes the clusters.  It
requires a distance matrix and an optional number of clusters.  The default
is 8.  Function includes a call to the dimensionReduce()and createFigure() functions.  
Also extracts the labels from the created clusters by selecting the 3 
ingredients closest to the centroid of each cluster.  These are the "canonical labels"


def executePhaseOne():
Simple function to be called in the main method.  This executes all of the
above methods in the proper order.

##############################################################################
##############################################################################
Bugs:

Again, note that main.py and phase1.py must be exectued from the 
"./project2_textAnalytics" directory to work.

The random state for kmeans seems to only apply if the main.py
is run again in the same session.  The labels will change from 
one session to the next


##############################################################################
##############################################################################
References:

Clustering Reference:
http://brandonrose.org/clustering

Data taken from:
Ahn, Y, et.al. "Flavor network and the principles of food pairing." 
Scientific Reports, 2011.
