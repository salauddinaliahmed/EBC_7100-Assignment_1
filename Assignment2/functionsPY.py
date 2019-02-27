from __future__ import print_function
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
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def calcPCA(pca_num_components, vector):
    reduced_data = PCA(n_components=pca_num_components).fit_transform(vector.todense())
    return reduced_data

def calcKnnElbow(elbowLimit, vector):
    distortions = []
    K = range(1,elbowLimit)
    Sum_of_squared_distances = []
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(vector)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
def plot3DScatter(kmeans, y_kmeans, vector):
    t1 = getTrace(vector[y_kmeans == 0, 0], vector[y_kmeans == 0, 1], vector[y_kmeans == 0, 2], s= 4, c='red', label = '1') #match with red=1 initial class
    t2 = getTrace(vector[y_kmeans == 1, 0], vector[y_kmeans == 1, 1], vector[y_kmeans == 1, 2], s= 4, c='black', label = '2') #match with black=3 initial class
    t3 = getTrace(vector[y_kmeans == 2, 0], vector[y_kmeans == 2, 1], vector[y_kmeans == 2, 2], s= 4, c='blue', label = '3') #match with blue=2 initial class
    x=vector[:,0]
    y=vector[:,1]
    z=vector[:,2]
    centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s= 8, c = 'yellow', label='Centroids')
    showGraph("Book Authors", "Bible", [min(x),max(x)], "Moby_Dick", [min(y),max(y)], "Parents", [min(z)-1,max(z)], [t1,t2,t3,centroids])
    
def knnErrorEval(vector, clustering_model, y_kmeans):
    labels_color_map = {
        0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1'
    }
    tsne_num_components = 3
    #Plot 3D scatter
    plot3DScatter(clustering_model, y_kmeans,vector)
    fig, ax = plt.subplots()
    for index, instance in enumerate(vector):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2, pca_comp_3 = vector[index]
        color = labels_color_map[y_kmeans[index]]
        ax.scatter(pca_comp_1, pca_comp_2, pca_comp_3, c=color)
    print("PCA")
    plt.show()
    
    # t-SNE plot
    embeddings = TSNE(n_components=tsne_num_components)
    Y = embeddings.fit_transform(vector)
    plt.scatter(Y[:, 0], Y[:, 1], Y[:, 2], cmap=plt.cm.Spectral)
    print("t-SNE")
    plt.show()
	
import seaborn as sns
def gmmErrorEval(gmm, vector):
    sns.distplot(vector, bins=20,kde=False,color=['g','r','b'])
    
from scipy.cluster.hierarchy import dendrogram, linkage
def aggErrorEval(aggCluster, vector):
    data = aggCluster.children_
    Z = linkage(data)
    dendrogram(Z)
    plotHeatMap(data)
    
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from numpy import arange

def plotHeatMap(data):
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
    

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)
def plotSill(vector):
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
        ax1.set_ylim([0, len(vector) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=0).fit(vector)
        cluster_labels = clusterer.fit_predict(vector)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(vector, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(vector, cluster_labels)

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
        ax2.scatter(vector[:, 0], vector[:, 1], marker='.', s=30, lw=0, alpha=0.7,
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

from sklearn.metrics import cohen_kappa_score
def getKappa(label, clusterModel):
    return cohen_kappa_score(list(label),clusterModel.labels_, weights="linear")

from sklearn.metrics import silhouette_score
def getSill(vector, clusterModel):
    return silhouette_score(vector, list(clusterModel.labels_))
    