#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

#from foreign_code import useful_function
from dataPreProcess import preprocess
from functionsPY import calcPCA
from functionsPY import knnErrorEval
from functionsPY import calcKnnElbow
from functionsPY import gmmErrorEval
from functionsPY import aggErrorEval
from functionsPY import getKappa
from functionsPY import getSill
from functionsPY import plotSill

from sklearn.cluster import KMeans

#static vars
numClusters = 3

#Pre process with 150 words per document
bow_vector, tfidf_vector, df_x, df_y = preprocess(150)

reduced_data_bow = calcPCA(numClusters, bow_vector)
reduced_data_tfidf = calcPCA(numClusters, tfidf_vector)

# KNN #########################################################################################

knn_bow_model = KMeans(n_clusters=numClusters,max_iter=300,precompute_distances="auto",n_jobs=-1).fit(reduced_data_bow)
knn_tfidf_model = KMeans(n_clusters=numClusters,max_iter=300,precompute_distances="auto",n_jobs=-1).fit(reduced_data_tfidf)

knn_bow = KMeans(n_clusters=numClusters,max_iter=300,precompute_distances="auto",n_jobs=-1).fit_predict(reduced_data_bow)
knn_tfidf = KMeans(n_clusters=numClusters,max_iter=300,precompute_distances="auto",n_jobs=-1).fit_predict(reduced_data_tfidf)

print("Elbow for Knn BOW:")
#calcKnnElbow(10, bow_vector)

print("Elbow for Knn TFIDF:")
#calcKnnElbow(10, tfidf_vector)

print("Knn for BOW:")
knnErrorEval(reduced_data_bow, knn_bow_model, knn_bow)
plotSill(reduced_data_bow)

print("Kappa")
print(getKappa(df_y, knn_bow_model))
# not working - print(getSill(df_y, knn_bow))

print("Knn for TFIDF")
knnErrorEval(reduced_data_tfidf, knn_tfidf_model, knn_tfidf)
plotSill(reduced_data_tfidf)

print("Kappa")
print(getKappa(df_y, knn_tfidf_model))
# not working - print(getSill(df_y, knn_tfidf))


# Gaussian Mixture - EM ########################################################################
color = ['g', 'r', 'b']
from sklearn.mixture import GaussianMixture

gmm_bow = GaussianMixture(n_components=numClusters).fit(reduced_data_bow)
gmm_tfidf = GaussianMixture(n_components=numClusters).fit(reduced_data_tfidf)
print("EM for BOW")
gmmErrorEval(gmm_bow, reduced_data_bow)

print("Kappa")

print("EM for TFIDF")
gmmErrorEval(gmm_tfidf, reduced_data_tfidf)

print("Kappa")

#Aggolomerative clustering ###############################################################################
from sklearn.cluster import AgglomerativeClustering

agg_cluster_bow = AgglomerativeClustering(n_clusters = numClusters).fit(reduced_data_bow)
agg_cluster_tfidf = AgglomerativeClustering(n_clusters = numClusters).fit(reduced_data_tfidf)

print("Agg for BOW")
aggErrorEval(agg_cluster_bow, reduced_data_bow)

print("Kappa")
print(getKappa(df_y, agg_cluster_bow))

aggErrorEval(agg_cluster_tfidf, reduced_data_tfidf)

print("Kappa")
print(getKappa(df_y, agg_cluster_tfidf))


#Consistency

