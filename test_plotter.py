# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:13:49 2018

@author: Ryan
"""
import matplotlib.pyplot as plt

def testPlotter(features_name, features, labels):
    
    indicesPOI=[i for i, x in enumerate(labels) if x==1]
    indicesNON=[i for i, x in enumerate(labels) if x==0]
 
    
    for ft, target in [features[i] for i in indicesNON]:
        plt.scatter( ft, target, color='r') 
    for ft, target in [features[i] for i in indicesPOI]:
        plt.scatter( ft, target, color='b') 


### labels for the legend
    plt.scatter(ft, target, color='b', label="POI")
    plt.scatter(ft, target, color='r', label="Non-POI")

### draw the regression line, once it's coded
    #try:
    #    plt.plot( feature_test, reg.predict(feature_test) )
    #except NameError:
    #    pass
    plt.xlabel(features_name[0])
    plt.ylabel(features_name[1])
    plt.legend()
    plt.show()
