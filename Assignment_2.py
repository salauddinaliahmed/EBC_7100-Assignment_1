#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:00:31 2019

@author: salauddinali
"""
import nltk
import pandas as pd
from nltk.corpus import gutenberg
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import k_means, hierarchical, AgglomerativeClustering,KMeans
from scipy.cluster.hierarchy import dendrogram, linkage  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cdist
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from collections import Counter
#Following libraries are used to supress warnings, the details can be found at
#https://blog.csdn.net/Homewm/article/details/84524558 
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

#Selecting the dataset
book_1 = gutenberg.raw('bible-kjv.txt').lower()
book_2 = gutenberg.raw('melville-moby_dick.txt').lower()
book_3 = gutenberg.raw('edgeworth-parents.txt').lower()
#Creates the list of all the three books
number_of_sentences = 45
book_list = [book_1, book_2, book_3]
master_list = [[],[],[]]
size_of_books = []
#Dividing each book into documents of 30 sentences. appending them to *master_list*. 
for book in book_list:
    n = 0
#Tokenzing sentences of each book
    book_sent = nltk.sent_tokenize(book)
#Identifying the number of documents we can create from each book based on length
    limit = int(len(book_sent)/number_of_sentences)
    size_of_books.append(limit)
    for i in range(limit):
        a = book_list.index(book)
        master_list[a].append(" ".join(book_sent[n:n+(number_of_sentences-1)]))
        n+=number_of_sentences 
#Seperating the 3 books.
complete_book_1 = (master_list[0])
complete_book_2 = (master_list[1])      
complete_book_3 = (master_list[2])
#Picking 200 samples from each book randomly without replacement
random_sample_book1 = random.sample(complete_book_1, 200)
random_sample_book2 = random.sample(complete_book_2, 200)
random_sample_book3 = random.sample(complete_book_3, 200)
#Tockenizing words for each document of each sample book
book1_words = [nltk.word_tokenize("".join(each_doc)) for each_doc in random_sample_book1]
book2_words = [nltk.word_tokenize("".join(each_doc)) for each_doc in random_sample_book2]
book3_words = [nltk.word_tokenize("".join(each_doc)) for each_doc in random_sample_book3]
stop_words = stopwords.words("english")
#Removing stop words, special characters and numbers from each document.
#clean book 1
clean_book1_words = []
for each_list in book1_words:
        clean_book1_words.append([each_word for each_word in each_list if not each_word in stop_words if each_word.isalpha()])
#clean book 2
clean_book2_words = []
for each_list in book2_words:
        clean_book2_words.append([each_word for each_word in each_list if not each_word in stop_words if each_word.isalpha()])
#clean book 3
clean_book3_words = []
for each_list in book3_words:
        clean_book3_words.append([each_word for each_word in each_list if not each_word in stop_words if each_word.isalpha()])
#Selecting 150 most common words.
most_common_book1 = []
for each_doc in clean_book1_words:
    a = Counter(each_doc[0]).most_common(150)
    most_common_book1.append(" ".join(i[0] for i in a))
most_common_book2 = []
for each_doc in clean_book2_words:
    a = Counter(each_doc[0]).most_common(150)
    most_common_book2.append(" ".join(i[0] for i in a))
most_common_book3 = []
for each_doc in clean_book3_words:
    a = Counter(each_doc[0]).most_common(150)
    most_common_book3.append(" ".join(i[0] for i in a))
#Once the samples are ready, we are converting it into dataframe for training the model.
sample_book1 = pd.DataFrame(most_common_book1)
sample_book2 = pd.DataFrame(most_common_book2)
sample_book3 = pd.DataFrame(most_common_book3)
#Converting Labels to integers
kjv_bible = 0
melville_moby_dick = 1
edgeworth_parents = 2
# Adding a new column to the sample books containing lables. 
sample_book1['Label'] = kjv_bible
sample_book2['Label'] = melville_moby_dick
sample_book3['Label'] = edgeworth_parents
#Labling columns of the dataframe
sample_book1.columns = ['Data', 'Label']
sample_book2.columns = ['Data', 'Label']
sample_book3.columns = ['Data', 'Label']
#Combining all the books to one and shuffling the samples
frames = [sample_book1, sample_book2, sample_book3]
complete_sample = pd.concat(frames, ignore_index=True)
complete_sample = shuffle(complete_sample)
complete_sample = complete_sample.reset_index(drop=True)
#--------------------end of pre-processing-------------------#
#Seperating Labels and Data from the complete sample.
df_x = (complete_sample['Data'])
df_y = (complete_sample['Label'])

#BOW Vectorization on the complete sample.
cv_bow = CountVectorizer(lowercase=False)
bow_vector= cv_bow.fit_transform(df_x)
#TFIDF Vectorization on the complete sample.
tfidf = TfidfVectorizer(lowercase=False)
tfidf_vector = tfidf.fit_transform(df_x)

#Elbow curve for K means
distortions = []
K = range(1,10)
Sum_of_squared_distances = []
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(bow_vector)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
#Clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(bow_vector)
print (Sum_of_squared_distances)

# EM
import seaborn as sns
from sklearn import mixture
X = bow_vector.toarray()
color = ['g']
#gmm = mixture.GaussianMixture(n_components=3, covariance_type='full',max_iter=1).fit(X)
#sns.distplot(X, bins=20,kde=False,color=color)

#Aggolomerative clustering
from sklearn.cluster import AgglomerativeClustering
agg_cluster = AgglomerativeClustering(n_clusters = 3).fit(X)

#Dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from numpy import arange

# Create the figure and set it's size.
fig = plt.figure(figsize=(50,150))

# Create the first subplot in the figure. 
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan= 1, colspan=1)

# Set to your favorite colormap.
cm = matplotlib.cm.jet

# Create array of random numbers.
X = data

# Create a linkage object.
linkmat = linkage(X)

# Make a dendrogram from the linkage object.
dendrogram(linkmat, truncate_mode="level")

# Use the x and y limits to set the aspect.
x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
#ax1.set_aspect((x1-x0)/(y1-y0))


# Remove the ticks on the x-axis. 
plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

# Create the second subplot.
ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan= 2, colspan=1)

labels = ["a", "b", "c"]

plt.xticks(arange(0.5, 7.5, 1))

plt.gca().set_xticklabels(labels)

plt.pcolor(X.T)

x0,x1 = ax2.get_xlim()
y0,y1 = ax2.get_ylim()

#ax2.set_aspect((x1-x0)/(y1-y0))

# Insert the color scale
plt.colorbar()
cb = plt.colorbar(ax=ax1)
cb.ax.set_visible(False)

# Make the vertical distance between plots equal to zero 
plt.subplots_adjust(hspace=0)

# Show the plot
plt.show()




#--------------------------------------------3Elbow curve
import math
"""
def getOptimalNumberOfClusters():
    wcss_values = buildWCSSValues()
    elbowIndex = getElbowPointIndex(wcss_values)
    return elbowIndex

def buildWCSSValues(X):
    from sklearn.cluster import KMeans
    print("Building WCSS Data...")
    wcss_values = []
    tmax_clusters = int(math.sqrt(len(bow_vector[0])).real)
    stepstr = ''
    print("Progression : ")
    for i in range(1, tmax_clusters) :
        print('\b'*len(stepstr))
        stepstr = str(i) + "/" + str(tmax_clusters - 1)
        print(stepstr)
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init = 10, random_state=0)
        kmeans.fit(X)
        wcss_values.append(kmeans.inertia_)
    return wcss_values

def getElbowPointIndex(wcss):
    curve = wcss
    nPoints = len(curve)
    allCoord = np.vstack((range(nPoints), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint

def showWCSSElbowGraph(wcss_values, elbowIndex):
    max_wcss = max(wcss_values)
    max_clusters = len(wcss_values)
    nb_clusters = np.arange(1, max_clusters+1, 1)
    wcss_r = np.array(wcss_values)/max_wcss
    nb_clusters_r = (1 * np.array(nb_clusters))/max_clusters
    plt.plot(nb_clusters_r, wcss_r)
    lx1=nb_clusters_r[0]
    ly1=wcss_r[0]
    lx2=nb_clusters_r[max_clusters - 1]
    ly2=wcss_r[max_clusters - 1]
    plt.plot([lx1, lx2], [ly1, ly2], c='green')
    coef = (ly2 - ly1)/(lx2 - lx1)
    plt.plot([nb_clusters_r[elbowIndex], 1], [wcss_r[elbowIndex], wcss_r[elbowIndex] - coef], c='red')
    plt.title('WCSS value according to the number of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS value')
    xticks = nb_clusters_r[0::1]
    xticks_lab = nb_clusters[0::1]
    plt.xticks(xticks, xticks_lab)
    ticks = np.arange(0, 1, 0.05)
    yticks = np.round(ticks * max_wcss) / max_wcss
    plt.yticks(yticks, (yticks*max_wcss).astype(int))
    plt.show()
"""
#---------- 3D plot
import plotly
import plotly.graph_objs as go
def getTrace(x, y, z, c, label, s=2):
    trace_points = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
    name=label
    )
    return trace_points;

def showGraph(title, x_colname, x_range, y_colname, y_range, z_colname, z_range, traces):
    layout = go.Layout(
    title=title,
    scene = dict(
    xaxis=dict(title=x_colname, range = x_range),
    yaxis=dict(title=y_colname, range = y_range),
    zaxis=dict(title=z_colname, range = z_range)
    )
    )

    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.plot(fig)
X=bow_vector
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter=15, n_init = 1, random_state=0)
y_kmeans = kmeans.fit_predict(X)
t1 = getTrace(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s= 4, c='red', label = '1') #match with red=1 initial class
t2 = getTrace(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s= 4, c='black', label = '2') #match with black=3 initial class
t3 = getTrace(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s= 4, c='blue', label = '3') #match with blue=2 initial class
x=X[:,0]
y=X[:,1]
z=X[:,2]
centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s= 8, c = 'yellow', label='Centroids')
showGraph("Book Authors", "Bible", [min(x),max(x)], "Moby_Dick", [min(y),max(y)], "Parents", [min(z)-1,max(z)], [t1,t2,t3,centroids])
print ("Done")


#K means validation
from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
"""X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility
"""
range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=0).fit(bow_vector)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

#kappa score.
from sklearn.metrics import cohen_kappa_score
print (cohen_kappa_score(list(df_y),agg_cluster.labels_, weights="linear"))
print (cohen_kappa_score(list(df_y),kmeans.labels_, weights="linear"))

#Consistency

#Silhouette 
from sklearn.metrics import silhouette_score
print (silhouette_score(bow_vector, list(agg_cluster.labels_)))
print (silhouette_score(bow_vector, list(kmeans.labels_)))