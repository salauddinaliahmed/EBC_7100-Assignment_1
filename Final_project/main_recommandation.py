#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:30:00 2019

@author: salauddinali
"""
from preprocessing import classify_preprocessing,clustering_preprocessing
from sklearn.svm import LinearSVC
from sklearn import svm, tree, metrics
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
from functions import calcPCA, pca_Plot, create_term_count_matrix, cluster_graph,most_common_graph,plot3DScatter,calcKnnElbow
from functions import clean_dict,stopwords_gen,recipe_similarity,closest_point,getSill
import pandas as pd
import json
from collections import Counter
from sklearn import metrics
from scipy.spatial import distance
from heapq import nsmallest

#For world map
import plotly.plotly as py #For World Map
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from flask import jsonify
from urllib.request import urlopen


#This is an API endpoint which is exposed at port 5200. This is consumed by the web application to fetch the recommendations
app = Flask(__name__)
api = Api(app)
CORS(app, support_credentials=True)

class main(Resource):

    @cross_origin()
    def get(self, reciepe_id):
        print("recieved request")        

        #Country_name_list
        country_names = ["Greece","Texas", "Philippines","India", "Jamaica", "Spain", "Italy", "Mexico","China", "Britian", "Thailand", "Vietnam", "United States", "Brazil", "France", "Japan","Ireland","Korea","Morocco", "Russia"]
        #number of cuisines
        numCuisine = 200 
        
        #data
        with urlopen('https://raw.githubusercontent.com/salauddinaliahmed/EBC_7100-Assignment_1/master/train.json') as data_file:    
            data = json.loads(data_file.read()) 
        
        #Classification data:
        complete_vector, x_train, x_test, y_train, y_test, df_sample, ingreds = classify_preprocessing(numCuisine)

        #Clustering data:
        cluster_vector, dictCuisineIngred, numCuisines, numIngred, cuisines, ingredients = clustering_preprocessing(data)
        
        #Mapping Cuisines to the country
        cuisine_country = dict(zip(cuisines, country_names))
        cuisines = list(dictCuisineIngred.keys())
        
        print(reciepe_id)
        print(type(reciepe_id))
        print(df_sample)
        selected_reciepe = df_sample.loc[df_sample["id"]==int(reciepe_id)]
        print(selected_reciepe.shape)
        print("selected_reciepe" , selected_reciepe)
        selected_cuisine = selected_reciepe.iloc[0]["cuisine"]
        print("selected_cuisine", selected_cuisine)
    
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
        
        #Clustering now, to show a better picture
        kmeans_model = KMeans(n_clusters=3,max_iter=300,precompute_distances="auto",n_jobs=-1)
        clus_kmeans = kmeans_model.fit(reduced_data_clean)
        clus_kmeans_predict = kmeans_model.fit_predict(reduced_data_clean)
        
        #This is the silhouette score.
        sil_score = getSill(reduced_data_clean, kmeans_model)
        print ("This is the Silhouette score: ", sil_score)


        #All cluster points.
        x_list, y_list = cluster_graph(new_clean_dict, cuisines, reduced_data_clean, clus_kmeans_predict)
        
        #recipe = input("Enter the id of the reciepe you like: ")
        
        #Calculating closest cluster point.
        index_of_cuisine = cuisines.index(selected_cuisine)
        close_cluster_index, x = closest_point(x_list,y_list,index_of_cuisine)
        
        #printing closest cuisine.
        print ("This is the closest cuisine to your selection: ", cuisines[close_cluster_index])
        
        #Selecting the most similar recipes.
        res_df = recipe_similarity(reciepe_id, close_cluster_index, df_sample, cuisines)
        
        print (res_df)
        #Converting the response to JSON and returning it to the web application
        return res_df.to_json(orient='records')
    
api.add_resource(main, '/reciepe/<reciepe_id>') # The API Endpoint

if __name__ == '__main__':
     app.run(port='5002')