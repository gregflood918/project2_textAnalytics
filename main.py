'''
Greg Flood
CS 5970 - Text Analytics

Project 2 - Phase 1
The goal of this project is to read in a list of paired
similarities for various types of food and perform 
clustering to explore similairites/differences between
the foods.

This is a simple main method to run the relevant code from
phase one.  Much more will be added for phase two, as it requires
a user-interface.  However, for phase1, all it requires is a call to
executePhaseOne() in phase1.py.  This performs k-means clustering and
visualizes the results.
import os, sys
lib_path = os.path.abspath(os.path.join('..','project1'))
sys.path.append(lib_path)
'''


from project2 import phase1
from project2 import phase2



class color:
#Color utility class.  This will provide some basic command line formatting
#and allow for important terms to be displayed in bol on a linux/unix OS
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   RED = '\033[91m'
   END = '\033[0m'

   
def searchMenu():
#User interface for the translation command line tool.  The user is allowed to
#select between 3 different options

    test = False
    while not test:
    
        print("\nPlease select " +color.UNDERLINE + "one" + color.END + 
        " of the following options:")
        print("Enter '1' for Help")
        print("Enter '2' to visualize ingredients via K-means clustering")
        print("Enter '3' to enter ingredients and search for an a similar recipe")
        print("Enter '0' to quit")
        choice = input ("Select: ")
        
        #User options
        if choice == "1":
            s = (color.BOLD + "\nDescription: " + color.END + "\nThis program will " 
                 "assist chefs in understanding food pairings and developing "
                 "creative menu otions.  The user is given the following choices:\n\n" + 
                 color.BOLD + color.RED +
                 "Option 1: " + color.END + "\nHelp menu \n\n" +
                 color.BOLD + color.RED +
                 "Option 2: " + color.END +
                 "\nPerforms kmeans clustering on the paired dataset from 'Flavor "
                 "Newtorks and the Principles of Food Pairing' which provides a list"
                 " of ingredient pairs along with the number of chemical compounds "
                 "shared in their molecular structure.  Clustering is performed by "
                 "creating an NxN (N= # of ingredients in data set) distance matrix "
                 "where element i,j is 1/(1+sim(i,j)) and sim(i,j) is the number of"
                 " ingredients shared by ingredients i and j.  However, the clustering "
                 "algorithm also accounts for indirect similarities, such as how many "
                 "compounds i an j share with ingredient k.  The visual display shows "
                 "the clusters mapped to distinct colors and displayed along the first "
                 "and second Principle Components, calculated using the PCA pacakge in "
                 "sci-kit learn. \n\n" + color.BOLD + color.RED +
                 "Option 3: " + color.END + "\nThe user will be prompted to enter "
                 "ingredient sequentially into the command line, entering '0' when complete. "
                 "The program will compare the entered ingredients to the yummly.json "
                 "dataset from 'Inferring Cuisine - Drug Interactions Using the Linked "
                 "Data Approach.'  The program will predict the most likely cuisine "
                 "style using 5 an ensemble classifier approach.  Additionallly, "
                 "the indices of the 5 most similar meals will be returned to the "
                 "user with the option to display the ingredients.  These simlarities "
                 "are computed via a 'Term Frequency - Inverse Document Frequency (tf-idf)' "
                 "approach.  For more technical details, please consult the README"
                 ".\n\n" + color.BOLD + color.RED +
                 "Option 4: \n" + color.END + "Exits the program\n\n")  
            print(s)
        elif choice == "2":
            print("\nComputing Clusters . . . \n")
            phase1.executePhaseOne()
             
        elif choice == "3":
            print("\nPlease enter ingredients one at a time and enter '0' when finished\n")
            ingredients = []
            num = 1
            user_input = input("Ingredient " + str(num) + ": ")
            while user_input != '0':
                ingredients.append(user_input) 
                num += 1
                user_input = input("Ingredient " + str(num) + ": ")
            phase2.run_option_three(ingredients)
        elif choice == "0":
            print("\nGoodbye")
            test = True
        else:
            print("\nInvalid Entry")



def main():
    # create database
    #phase1.executePhaseOne()
    searchMenu()
    
    
if __name__ == '__main__':
    main()