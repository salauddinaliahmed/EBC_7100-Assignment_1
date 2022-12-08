#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:30:00 2019

@author: salauddinali
"""
from preprocessing import classify_preprocessing,clustering_preprocessing
from sklearn.svm import LinearSVC
from sklearn import svm, tree, metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_samples, silhouette_score, homogeneity_completeness_v_measure, homogeneity_score, 
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from functions import evaluate_clusters, calcPCA, pca_Plot, create_term_count_matrix, cluster_graph, cloud_words, most_common_graph, plot3DScatter,calcKnnElbow
from functions import clean_dict,stopwords_gen,recipe_similarity,closest_point,getKappa,getSill,world_map
import pandas as pd
import json
from collections import Counter
from sklearn import metrics
from scipy.spatial import distance
from heapq import nsmallest
from functions import plotSill
#For world map
import plotly.plotly as py #For World Map
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from urllib.request import urlopen

#Country_name_list
country_names = ["Greece","Texas", "Philippines","India", "Jamaica", "Spain", "Italy", "Mexico","China", "Britian", "Thailand", "Vietnam", "United States", "Brazil", "France", "Japan","Ireland","Korea","Morocco", "Russia"]
#number of cuisines
numCuisine = 400
#data
with urlopen('https://raw.githubusercontent.com/salauddinaliahmed/EBC_7100-Assignment_1/master/train.json') as data_file:    
    data = json.loads(data_file.read()) 

#for recommendation
complete_vector, x_train, x_test, y_train, y_test, df_sample, ingreds = classify_preprocessing(numCuisine)

#Clustering data:
cluster_vector, dictCuisineIngred, numCuisines, numIngred, cuisines, ingredients = clustering_preprocessing(data)

#Mapping Cuisines to the country
cuisine_country = dict(zip(cuisines, country_names))
cuisines = list(dictCuisineIngred.keys())


#Taking user input:
print ("List of countries: ", country_names)
selected_country = input("Please select your nationality:")


#Priting out top 20 recipies across the selected country.
for k, i in cuisine_country.items():
    if i == selected_country:
        selected_cuisine = k

#Printing the list of recipies along the cuisines. 
print (df_sample[df_sample.cuisine==selected_cuisine].head(10))

#Asking the user for a reciepe
print ("------------------------------------")
print ("Here comes the good stuff")

#Calculating Stopwords
stop_words, stop_tuple = stopwords_gen(dictCuisineIngred)

#Plotting Stopwords.
most_common_graph(stop_tuple)

#New Bag of words after removing stopwords
new_total_ingredient_set, new_total_ingredients, new_clean_dict = clean_dict(dictCuisineIngred,stop_words)

#Calculating clusters
custom_bow_clean = create_term_count_matrix(new_clean_dict, numCuisines, new_total_ingredients, cuisines, new_total_ingredient_set)

#TFIDF Transfrom the custom_bow
tfidf_v = TfidfTransformer().fit_transform(custom_bow_clean)

#Making the vector dense to be passed into clustering algorithm
dense_tfidf = tfidf_v.todense()
#Reduced PCA 
reduced_data_clean = calcPCA(2, dense_tfidf)

#Plotting the elbow curve.
calcKnnElbow(5, reduced_data_clean)

#Clustering now, to show a better picture
kmeans_model = KMeans(n_clusters=3,max_iter=300,precompute_distances="auto",n_jobs=-1)
clus_kmeans = kmeans_model.fit(reduced_data_clean)
clus_kmeans_predict = kmeans_model.fit_predict(reduced_data_clean)

#All cluster points.
x_list, y_list = cluster_graph(new_clean_dict, cuisines, reduced_data_clean, clus_kmeans_predict)

recipe = input("Enter the id of the reciepe you like: ")

#Calculating closest cluster point.
index_of_cuisine = cuisines.index(selected_cuisine)
close_cluster_index, sim_country = closest_point(x_list,y_list,index_of_cuisine)

#printing closest cuisine.
print ("This is the closest cuisine to your selection: ", cuisines[close_cluster_index])

#Selecting the most similar recipes.
res_df = recipe_similarity(recipe, close_cluster_index, df_sample, cuisines)

print (res_df)

#Clustering overall data. For Scores.
kmeans_model_org = KMeans(n_clusters=20,max_iter=300,precompute_distances="auto",n_jobs=-1)
clus_kmeans_org = kmeans_model_org.fit(x_train)
clus_kmeans_predict_org = kmeans_model_org.fit_predict(x_train)
calcKnnElbow(22, x_train)

#LabelEncoding
le = preprocessing.LabelEncoder()
t_ytrain = le.fit_transform(y_train)

kappa_score = getKappa(t_ytrain, kmeans_model_org)
print ("This is the Kappa Score: ", kappa_score)

sil_score = getSill(x_train, kmeans_model_org)
print ("This is the Silhouette score: ", sil_score)

#The clusters are overlapping and highly inconsistence due to alot of overlapping ingredients. 


#Calculating country similarity
act_index = cuisines.index(selected_cuisine)
sim_c = metrics.pairwise.cosine_similarity(reduced_data_clean,reduced_data_clean[act_index].reshape(1, -1))
sim_cc = [float(each_it[0]) for each_it in sim_c]

cloud_words(new_clean_dict)
world_map(sim_cc, country_names, selected_country)

#plotSill (dense_tfidf.todense())

print ("This is the Rand index: ",metrics.adjusted_rand_score(t_ytrain,kmeans_model_org.labels_) )
evaluate_clusters (t_ytrain,kmeans_model_org.labels_)