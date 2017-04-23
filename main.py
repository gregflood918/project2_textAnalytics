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
            s = (color.BOLD + "\nDescription: " + color.END + "\nThis program will search "
                 "for a Latin or English word in the "
                 "following Latin collections:\n\n" + 
                 "\nSearches for a user supplied Latin word in all of the "
                 "aforemention corpuses and returns passages contain the supplied "
                 "word, along with the Title, Book, Chapter, Verse, and link for "
                 "the passage.   Additionally, a barplot displaying the frequency of "
                 "the word usage across all 8 collections will be returned.\n\n" + color.BOLD +
                 "Option 3: " + color.END + "\nSearches for a user supplied English word in all of the "
                 "aforemention corpuses and returns passages contain the supplied "
                 "word, along with the Title, Book, Chapter, Verse, and link for "
                 "the passage.  The English term is translated to Latin via the "
                 "mymemory.translated.net translation API, and the translation "
                 "will be given alongside the prevoius results. Additionally, a "
                 "barplot displaying the frequency of the word usage across all "
                 "8 collections will be returned.\n\n" + color.BOLD + 
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