
Project 2 - Phase One
Greg Flood
gflood@ou.edu


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
other observation at each merge step.

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
algorithm is initialized to 101, so the result will be the same every simulation.
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


We can see that these labels make some intuitive sense, as we see beverages,
sour fruits, meats, and peppers in their own distinct groups.


##############################################################################
##############################################################################

Language:
Python 3

##############################################################################
##############################################################################

Testing:
Since there is no user-facing interface for phase1, there are no tests modules. Also,
the project description does not say anything about creating tests.


##############################################################################
##############################################################################

Instructions:

To run the program, simply navigate to the project2_textAnalytics folder and
execute the following command:

python3 main.py

main.py contains a call to the function executePhaseOne(), which executes
the relevant functions of phase 1 in the proper sequential order.


project2_textAnalytics/
        project2/
                phase1.py
                __init__.py
        README
        setup.py
        setup.cfg
        requirements.txt
        main.py
	docs/
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


##############################################################################
##############################################################################
References: