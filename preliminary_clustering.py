#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:34:34 2019

@author: jiaxichen
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import functionsPY
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


data = pd.read_csv("/Users/jiaxichen/Desktop/Assignment 3/epicurious-recipes-with-rating-and-nutrition/epi_r.csv")



recipe_title = data["title"]
#cluster_recipies = KMeans(n_clusters=5,max_iter=300,precompute_distances="auto",n_jobs=-1).fit_predict(data)

data = data.drop(columns = 'title')

print ("Recipe data", recipe_title.head())

print ("Data", data.head())
data = data.fillna(0)

print("Elbow for Knn BOW:")
#functionsPY.calcKnnElbow(20, data)


cv_bow = CountVectorizer(lowercase=False)
data = cv_bow.fit_transform(data)
data = data.todense()
kmeans_model = KMeans(n_clusters=3,max_iter=300,precompute_distances="auto",n_jobs=-1).fit(data)
y_kmeans = KMeans(n_clusters=3,max_iter=300,precompute_distances="auto",n_jobs=-1).fit_predict(data)

functionsPY.knnErrorEval(data,kmeans_model,y_kmeans)

