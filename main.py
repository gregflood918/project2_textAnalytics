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
'''

import project2
from project2 import phase1


def main():
    # create database
    phase1.executePhaseOne()
    
if __name__ == '__main__':
    main()