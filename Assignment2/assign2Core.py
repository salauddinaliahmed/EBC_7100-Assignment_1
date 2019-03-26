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
from functionsPY import getKappa, agg_cluster_graph
from functionsPY import getSill, pca_Plot, plot_eval_comp,plot_eval_comp
from functionsPY import plotSill, evaluate_clusters, getKappa_gmm 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#static vars
numClusters = 3

#Preprocessing and error evaluation
words150 = 150
#words10 = 10
#coll = "colloc"


#Pre process with 150 words per document
bow_vector, tfidf_vector, df_x, df_y = preprocess(words150)


reduced_data_bow, var_ratio = calcPCA(numClusters, bow_vector)
reduced_data_tfidf, var_ratio = calcPCA(numClusters, tfidf_vector)
#reduced_data_bow = bow_vector.todense()
#reduced_data_tfidf = tfidf_vector.todense()

#print ("Vairance for BOW PCA for 3 prinicpal components", var_ratio)
pca_Plot(bow_vector)
print (reduced_data_bow.shape)

print ("TFIDF")
#print ("Vairance for TFIDF PCA for 3 prinicpal components", var_ratio)
print (reduced_data_tfidf.shape)
pca_Plot(tfidf_vector)




# KNN #########################################################################################

kmeans_bow_model = KMeans(n_clusters=numClusters,max_iter=300,precompute_distances="auto",n_jobs=-1).fit(reduced_data_bow)
kmeans_tfidf_model = KMeans(n_clusters=numClusters,max_iter=300,precompute_distances="auto",n_jobs=-1).fit(reduced_data_tfidf)

#print ("Unit Test::::::")
#word = 'That he would grant you, according to the riches of his glory to be strengthened with might by his Spirit in the inner man; That Christ may dwell in your hearts by faith; that ye, being rooted and grounded in love.'
word = [df_x[1]]
cv_bow = CountVectorizer(lowercase=False)
#print (bow_vector)
#kmeans_bow = kmeans_bow_model.predict(bow_vector.toarray())
#print (kmeans_bow)
predict_word = cv_bow.fit_transform(word)

print ("This is the end of unit testing.")
kmeans_bow = KMeans(n_clusters=numClusters,max_iter=300,precompute_distances="auto",n_jobs=-1).fit_predict(reduced_data_bow)
kmeans_test = KMeans(n_clusters=numClusters,max_iter=300,precompute_distances="auto",n_jobs=-1).fit(reduced_data_bow)
 
kmeans_tfidf = KMeans(n_clusters=numClusters,max_iter=300,precompute_distances="auto",n_jobs=-1).fit_predict(reduced_data_tfidf)
print ('This is the predicted cluster of the part :',kmeans_test.predict(predict_word))

print("Elbow for Knn BOW:")
calcKnnElbow(8, reduced_data_bow)

print("Elbow for Knn TFIDF:")
calcKnnElbow(8, reduced_data_tfidf)


knnErrorEval(reduced_data_bow, kmeans_bow_model, kmeans_bow)
plotSill(reduced_data_bow)
kappa_kmeans_bow = getKappa(df_y, kmeans_bow_model)
print ("Evaluation scores for BOW Kmeans")
print ("Evaluation scores")
print("Kappa score: ",kappa_kmeans_bow)
homo_kmeans_bow, compl_kmeans_bow, vm_kmeans_bow, a_rand_kmeans_bow, a_mutual_kmeans_bow = evaluate_clusters(df_y, kmeans_bow)
sil_kmeans_bow = silhouette_score(reduced_data_bow, kmeans_bow_model.labels_)
print("Silhouette score: ",sil_kmeans_bow)


knnErrorEval(reduced_data_tfidf, kmeans_tfidf_model, kmeans_tfidf)
plotSill(reduced_data_tfidf)
print("Evaluation scores for KMeans TFIDF")
print ("Evaluation scores ")
kappa_kmeans_tfidf = getKappa(df_y, kmeans_tfidf_model)
print("Kappa score: ",kappa_kmeans_tfidf)
homo_kmeans_tfidf, compl_kmeans_tfidf, vm_kmeans_tfidf, a_rand_kmeans_tfidf, a_mutual_kmeans_tfidf = evaluate_clusters(df_y, kmeans_tfidf)
sil_kmeans_tfidf = silhouette_score(reduced_data_tfidf, kmeans_tfidf_model.labels_)
print("Sillhouette score:",sil_kmeans_tfidf)





# Gaussian Mixture - EM ########################################################################
print("EM clustering")
color = ['g', 'r', 'b']
from sklearn.mixture import GaussianMixture

#Vectorization
gmm_bow = GaussianMixture(n_components=numClusters).fit(reduced_data_bow)
gmm_tfidf = GaussianMixture(n_components=numClusters).fit(reduced_data_tfidf)

#print ("GMM Contours")
#contour_gmm(gmm_bow, reduced_data_bow, df_y)

print("EM for BOW")
print ("Cluster formation for EM for BOW")

labels = gmm_bow.predict(reduced_data_bow)
gmm_labels = gmm_bow.fit_predict(reduced_data_bow)

kappa_gmm_bow = getKappa_gmm(df_y, gmm_labels)
print("Kappa for GMM: ",kappa_gmm_bow)

sil_gmm_bow = silhouette_score(reduced_data_tfidf, labels) 
print("Sillhouette score for GMM(EM) BOW", sil_gmm_bow)

probs = gmm_tfidf.predict_proba(reduced_data_bow)
size = 50 * probs.max(1) ** 2

plt.scatter(list(df_y), reduced_data_bow[:, 1] ,c=labels, s=size, cmap='viridis')
plt.title("GMM BOW")
plt.show()

homo_gmm_bow, compl_gmm_bow, vm_gmm_bow, a_rand_gmm_bow, a_mutual_gmm_bow = evaluate_clusters(df_y, gmm_labels)

print("EM for TFIDF")
#gmmErrorEval(gmm_tfidf, reduced_data_tfidf)
print("Cluster formation of EM for TFIDF")
labels = gmm_tfidf.predict(reduced_data_tfidf)
gmm_labels = gmm_tfidf.fit_predict(reduced_data_tfidf) 
kappa_gmm_tfidf = getKappa_gmm(df_y, gmm_labels)
print ("Evaluation scores")
print("Kappa score: ", kappa_gmm_tfidf)
homo_gmm_tfidf, compl_gmm_tfidf, vm_gmm_tfidf, a_rand_gmm_tfidf, a_mutual_gmm_tfidf= evaluate_clusters(df_y, gmm_labels)
sil_gmm_tfidf = silhouette_score(reduced_data_tfidf, labels)
print ("Silhouette score: ", sil_gmm_tfidf)

# Plotting a scatterplot for tfidf GMM
probs = gmm_tfidf.predict_proba(reduced_data_tfidf)
size = 50 * probs.max(1) ** 2


plt.scatter(list(df_y),reduced_data_tfidf[:, 1], c=labels, s=size)
plt.title("GMM TFIDF")
plt.show()






#Aggolomerative clustering ###############################################################################
print ("Aggolomerative Clustering")
from sklearn.cluster import AgglomerativeClustering

agg_cluster_bow = AgglomerativeClustering(n_clusters = numClusters).fit(reduced_data_bow)
agg_cluster_tfidf = AgglomerativeClustering(n_clusters = numClusters).fit(reduced_data_tfidf)
agg_bow = agg_cluster_bow.fit_predict(reduced_data_bow)
agg_tfidf = agg_cluster_tfidf.fit_predict(reduced_data_tfidf)
print("Agg for BOW")
aggErrorEval(agg_cluster_bow, reduced_data_bow)


#Graph/cluster for agg
print ("Cluster graph for aggolomerative clustering BOW")
agg_cluster_graph(reduced_data_bow,df_y,agg_cluster_bow, "BOW")

print ("Cluster graph for aggolomerative clustering TFIDF")
agg_cluster_graph(reduced_data_tfidf,df_y,agg_cluster_tfidf, "TFIDF")




### Evaluation and comparison ################################################################
print ("Error evaluations for Aggolomerative BOW")
kappa_agg_bow = getKappa(df_y, agg_cluster_bow)
print ("Evaluation Scores ")
print("Kappa score: ", kappa_agg_bow)
homo_agg_bow, compl_agg_bow, vm_agg_bow, a_rand_agg_bow, a_mutual_agg_bow = evaluate_clusters(df_y, agg_bow)
sill_agg_bow = silhouette_score(reduced_data_bow, agg_cluster_bow.labels_)
print("Sillhouette score:", sill_agg_bow)

print ("Error evaluations for Aggolomerative TFIDF")
aggErrorEval(agg_cluster_tfidf, reduced_data_tfidf)


print ("Error evaluations for Aggolomerative TFIDF")
kappa_agg_tfidf = getKappa(df_y, agg_cluster_tfidf)
print ("Evaluation Scores ")
print ("Kappa score: ", kappa_agg_tfidf)
homo_agg_tfidf, compl_agg_tfidf, vm_agg_tfidf, a_rand_agg_tfidf, a_mutual_agg_tfidf = evaluate_clusters(df_y, agg_tfidf)
sill_agg_tfidf = silhouette_score(reduced_data_tfidf, agg_cluster_tfidf.labels_)
print("Sillouette score: ", sill_agg_tfidf )

#Homogeneity score comparison for all models. 

homo_comparision_bow = [homo_kmeans_bow,homo_gmm_bow,homo_agg_bow]
homo_comparision_tfidf = [homo_kmeans_tfidf,homo_gmm_tfidf,homo_agg_tfidf] 
plot_eval_comp(homo_comparision_bow, homo_comparision_tfidf, "Homogeneity Score")

#completeness score comparison for all models.
comp_comparison_bow = [compl_kmeans_bow, compl_gmm_bow ,compl_agg_bow]
comp_comparison_tfidf = [compl_kmeans_tfidf, compl_gmm_tfidf, compl_agg_tfidf]
plot_eval_comp(comp_comparison_bow, comp_comparison_tfidf, "Consistency Score")

#VM score
comp_vm_bow = [vm_kmeans_bow, vm_gmm_bow ,vm_agg_bow] 
comp_vm_tfidf = [vm_kmeans_tfidf, vm_gmm_tfidf, vm_agg_tfidf] 
plot_eval_comp(comp_vm_bow, comp_vm_tfidf, "VM score")

#Random Aggregate 
arand_agg_bow = [a_mutual_kmeans_bow,a_mutual_agg_bow,a_mutual_agg_bow] 
arand_agg_tfidf = [a_mutual_kmeans_tfidf, a_mutual_gmm_tfidf, a_rand_agg_tfidf] 
plot_eval_comp(arand_agg_bow, arand_agg_tfidf, "Random Aggregate Score")

#Mutual Aggregate
magg_ma_bow = [a_mutual_kmeans_bow,a_mutual_gmm_bow,a_mutual_agg_bow] 
magg_ma_tfidf = [a_mutual_kmeans_tfidf, a_mutual_gmm_tfidf, a_mutual_agg_tfidf] 
plot_eval_comp(magg_ma_bow, magg_ma_tfidf, "Mutual Aggregate")

#Kappa score
kappa_bow = [kappa_kmeans_bow, kappa_gmm_bow, kappa_agg_bow]
kappa_tfidf = [kappa_kmeans_tfidf, kappa_gmm_tfidf, kappa_agg_tfidf]
plot_eval_comp(kappa_bow,kappa_tfidf, "Kappa Score")

#Silhouette score
sil_bow = [sil_kmeans_bow, sil_gmm_bow, sill_agg_bow]
sil_tfidf = [sil_kmeans_tfidf, sil_gmm_tfidf, sill_agg_tfidf]
plot_eval_comp(sil_bow,sil_tfidf, "Silhouette Score")

