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
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy import *
from scipy.spatial import distance
from heapq import nsmallest   
from sklearn import metrics
#For world map
import plotly.plotly as py #For World Map
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import gutenberg
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, tree
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from collections import Counter
import seaborn as sn


#---------- 3D plot
import plotly
import plotly.graph_objs as go
def getTrace(x, y, z, c, label, s=[1,2,4,5,6]):
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
    reduced_data = PCA(n_components=pca_num_components).fit_transform(vector)
    return reduced_data

#PCA plot, but we are not using it.
def pca_Plot(vector):
    #scaled_x = MinMaxScaler().fit_transform(vector.todense())
    #print ("Scaled Matrix")
    pca_comp = PCA().fit(vector)
    #Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca_comp.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Cuisine dataset PCA scores')
    plt.show()    
    
def create_term_count_matrix(dictionary, numCuisines, numIngred, cuisines, ingredients):
    termCountMatrix = np.zeros((numCuisines,numIngred))
    i = 0
    
    for cuisine in cuisines:
        ingredientsPerCuisine = dictionary[cuisine]

        for ingredient in ingredientsPerCuisine:
            j = ingredients.index(ingredient) #in order to know which column to put the term count in, we will ago according to the terms' order in the ingredients array
            termCountMatrix[i,j] += 1
        i += 1

    return termCountMatrix

def calcKnnElbow(elbowLimit, vector):
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
    t1 = getTrace(vector[y_kmeans == 0, 0], vector[y_kmeans == 0, 1], vector[y_kmeans == 0, 2], s= 4, c='red', label = 'Center1') #match with red=1 initial class
    t2 = getTrace(vector[y_kmeans == 1, 0], vector[y_kmeans == 1, 1], vector[y_kmeans == 1, 2], s= 4, c='black', label = 'Center2') #match with black=3 initial class
    t3 = getTrace(vector[y_kmeans == 2, 0], vector[y_kmeans == 2, 1], vector[y_kmeans == 2, 2], s= 4, c='blue', label = 'Center3') #match with blue=2 initial class
    t4 = getTrace(vector[y_kmeans == 3, 0], vector[y_kmeans == 3, 1], vector[y_kmeans == 3, 2], s= 4, c='yellow', label = 'Center4')
    t5 = getTrace(vector[y_kmeans == 4, 0], vector[y_kmeans == 4, 1], vector[y_kmeans == 4, 2], s= 4, c='orange', label = 'Center5')
    print ("This is the value for plot3d:", vector[y_kmeans == 0, 0])
    x=vector[:,0]
    y=vector[:,1]
    z=vector[:,2]
    centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s= 8, c = 'yellow', label='Centroids')
    showGraph("Cuisines", "PCA1", [min(x),max(x)], "PCA2", [min(y),max(y)], "PCA3", [min(z)-1,max(z)], [t1,t2,t3,t4,t5,centroids])
    

from wordcloud import WordCloud
stop = set(stopwords.words('english'))

def cloud_words(dictCuisineIngred):
    for cuisine, val in dictCuisineIngred.items():
        text_list = (dictCuisineIngred[cuisine])
        text = " ".join(text_list)
        
        wordcloud = WordCloud(max_font_size=None, background_color='white',
                              width=1200, height=1000).generate(text)
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud)
        plt.title('Top ingredients in '+ cuisine+' food')
        plt.axis("off")
        plt.show()
    
def cluster_graph(dictCuisineIngred, cuisines, reduced_data, labels):
    i = 0 
    j = 0 
    
    effect_on_cluster = [0 for cuisine in cuisines]
    for cuisineA in cuisines:  

        A_intersection = 0
        numInClusterBesidesA = 0
        setA = set(dictCuisineIngred[cuisineA])
        setB_forA = []
        j = 0
        
        for cuisineB in cuisines:
            if cuisineB != cuisineA: # if it is A itself - we obviously wouldn't want this (will be exactly 1)
                if labels[j] == labels[i]: #determines if then they are both in the same cluster
                    setB_forA.extend(set(dictCuisineIngred[cuisineB]))
                    numInClusterBesidesA += 1
            j += 1
        
        A_intersection = len(set(setA & set(setB_forA))) / float(len(set(setA.union(setB_forA))))
        effect_on_cluster[i] = A_intersection
           
        i += 1
        
    import matplotlib.pyplot as plt

    rdata = reduced_data
    i=0
    figureRatios = (15,20)
    x = []
    y = []
    color = []
    area = []
    
    #creating a color palette:
    colorPalette = ['#009600','#2980b9', '#ff6300','#2c3e50', '#660033']
    # green,blue, orange, grey, purple
    
    plt.figure(1, figsize=figureRatios)
    
    for data in rdata:
        x.append(data[0]) 
        y.append(data[1])  
        color.append(colorPalette[labels[i]]) 
        area.append(effect_on_cluster[i]*27000) # magnifying the bubble's sizes (all by the same unit)
        # plotting the name of the cuisine:
        text(data[0], data[1], cuisines[i], size=10.6,horizontalalignment='center', fontweight = 'bold', color='w')
        i += 1
    
    plt.scatter(x, y, c=color, s=area, linewidths=2, edgecolor='w', alpha=0.80) 
    
    plt.axis([-0.45,0.65,-0.55,0.55])
    plt.axes().set_aspect(0.8, 'box')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.axis('off') # removing the PC axes
    plt.show()
    return x,y 

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

def most_common_graph(common_stuff):
    x = []
    y = []
    for each_stuff in common_stuff:
        x.append(each_stuff[0])
        y.append(each_stuff[1])
    plt.figure(figsize=(15,15))
    plt.bar(x,y, align='center', alpha=0.5)
    plt.xticks(x)
    plt.ylabel('Frequency')
    plt.title('Ingredients')
    plt.show()
    return x
    
#Scores Kappa and Silhouette
from sklearn.metrics import cohen_kappa_score
def getKappa(label, clusterModel):
    return cohen_kappa_score(list(label),clusterModel.labels_, weights="linear")

def getSill(vector, clusterModel):
    return silhouette_score(vector, list(clusterModel.labels_))
  
# Cleaning the dictoinary
def clean_dict(dictionary, stops):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    dummy_list = []
    new_list = []
    stopwords_clean = []
    for each_item in values:
    
        new_list = []
        for each_dummy in each_item:
            if each_dummy not in stops:
                dummy_list.append(each_dummy)
                new_list.append(each_dummy)
        stopwords_clean.append(new_list)    
    
    new_total_ingredient_set = list(set(dummy_list))
    new_total_ingredients = len(new_total_ingredient_set)
    new_clean_dict = dict(zip(keys, stopwords_clean))
    return new_total_ingredient_set, new_total_ingredients, new_clean_dict

#Cleaning the classification dict:
def clean_class(only_ingred, stop_w):
    new_list = []
    clean_list = []
    dummy_list = []
    new_list = []
    stopwords_clean = []
    for each_list in only_ingred:
        for each_item in each_list[0]:
            if each_item.strip(" ") not in stop_w:
                new_list.append(each_item.strip(" "))
                
        stopwords_clean.append(new_list)
        new_list = []
    return stopwords_clean

#Plotting stopwords
def stopwords_gen(dictionary):
    ingredients_percuisine = []
    ingredients_percuisine = dictionary.values()
    list_of_ingred = []
    for each_ingred in ingredients_percuisine:
        list_of_ingred.append(each_ingred)
    
    all_ingred = ",".join(inner for i in list_of_ingred for inner in i) 
    each_ingred = all_ingred.split(',')
     
    most_common_ingred = Counter(list(each_ingred)).most_common(10)
    stp_wrds = []
    for each_tuple in most_common_ingred:
        stp_wrds.append(each_tuple[0])
    return stp_wrds, most_common_ingred

#finding closest cuisine structure.
def closest_point(x_stuff, y_stuff, index):
    sim_scores =[]
    count = 0 
    for count in range(len(x_stuff)):
        a = (x_stuff[count], y_stuff[count])
        b = (x_stuff[index], y_stuff[index])
        sim_scores.append(distance.euclidean(a, b))
    
    value_similar = nsmallest(2, sim_scores)[-1]
    true_index = sim_scores.index(value_similar)
    return true_index, sim_scores

#Calculating similarity score among recipes.
def recipe_similarity(r_id,index_cuisine,df_sample, cuisines):
    #This id should be the cuisine id
    fetched_similar = df_sample[df_sample["cuisine"] == str(cuisines[index_cuisine])]
    user_ingredients = df_sample[df_sample.id == r_id]
    user_data = user_ingredients[["ingredients_clean_string", "id"]]
    fetched_data = fetched_similar[["ingredients_clean_string", "id"]]
    combined_data = fetched_data.append(user_data, ignore_index=True)
    combined_data_tfidf = TfidfVectorizer(lowercase=False).fit_transform(combined_data["ingredients_clean_string"])
    
    #fetched_tfidf = TfidfVectorizer(lowercase=False).fit_transform(fetched_similar["ingredients_clean_string"])
    similarity_array = metrics.pairwise.cosine_similarity(combined_data_tfidf, combined_data_tfidf[-1])
    user_cuisine_scores = dict(zip(combined_data['id'], similarity_array))
    test_dict = sorted(list(user_cuisine_scores.values()))[-4:-1] 
    id_in=[]
    v_in =[]
    for k, v in user_cuisine_scores.items():
        if v in test_dict:
            id_in.append(k)
            v_in.append(v)
    new_dict = dict(zip(id_in,v_in))    
    resultant_df = pd.DataFrame(columns=['recipe_id', 'ingredients', 'cuisine', 'similarity_score'])
    for k,val in new_dict.items():
        recipe = k
        temp = df_sample[df_sample["id"]==k]
        cuisine = temp["cuisine"]
        ingredients = temp["ingredients"]
        sim_sc = float(val[0])
        sim_sc = round(sim_sc*100, 3)
        resultant_df = resultant_df.append({'recipe_id': recipe, 'ingredients': ingredients, 'cuisine': cuisine, 'similarity_score':sim_sc}, ignore_index=True)
    
    return resultant_df
    
    #print ("This is the similar cuisine",fetched_data.loc[fetched_data['id'] == id_in])
    
#Function to display world map
def world_map(sim_cc, country_names, s_country):
    data = dict(type = 'choropleth', 
               locations = country_names,
               locationmode = 'country names',
               z = sim_cc, 
               text = country_names,
               colorbar = {'title':'Similarity'})
    layout = dict(title = 'Similarity of cuisines to your country, '+ s_country, 
                 geo = dict(showframe = False, 
                           projection = {'type': 'mercator'}))
    choromap3 = go.Figure(data = [data], layout=layout)
    plot(choromap3)

########### JIAXI CODE  ###############

#---------------------Defining the functions--------------------#
#Function to plot accuracies for each vector
def accuracy_plot(k_folds, tfidf_score, classifier_used,title):    
    plt.plot(list(range(1,k_folds+1)),list(tfidf_score), 'r--')
    plt.axis([1, k_folds, 0.00, 1.00])
    legend = mpatches.Patch(color='red', label=title)
    plt.legend(handles=[legend])
    plt.title(classifier_used)
    plt.xlabel('K folds')
    plt.ylabel('Accuracy')
    plt.show()

#Validation curve for SVM
#This code snippet is taken from
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html and modified
def validation_curve_graph(complete_vector, df_y, clf, parameter, title):
    X, y = complete_vector, df_y
    cv = StratifiedKFold(10)
    param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        clf, X, y, param_name= parameter, param_range=param_range,
        cv=cv, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title(title)
    plt.xlabel(parameter)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()    
#Function to plot learning curve wrt train size. The following function was taken from
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html and modified 
def learning_curve_graph(clf, word_vector, df_y):   
    train_sizes, train_scores, valid_scores = learning_curve(clf, word_vector, df_y, train_sizes=[50, 100, 150, 200, 250, 300, 350, 400, 500], cv=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.xlabel("Sample Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()
def generate_error_eval(classfier, train_vector, target_variables,target_names, k_folds):
   error_evaluation = cross_val_predict(estimator=classfier, 
                    X=train_vector, 
                    y=target_variables, 
                    cv=k_folds)
   confu_mat = confusion_matrix(target_variables, error_evaluation)
   print(classification_report(target_variables, error_evaluation, target_names=target_names))
   return confu_mat

def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       print("Normalized confusion matrix")
   else:
       print('Confusion matrix, without normalization')

   plt.figure(figsize = (20,20))
   sn.heatmap(cm, annot=True)
   plt.title(title)
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes, rotation=90)
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.tight_layout()
def evaluate_clusters(labels, clusters):
    print ("Homogeneity Score: ",metrics.homogeneity_score(labels, clusters))
    print ("Completeness Score :",metrics.completeness_score(labels, clusters))
    print ("V_measure: ",metrics.v_measure_score(labels, clusters))
    print ("Adjusted Random Score: ",metrics.adjusted_rand_score(labels, clusters))
    print ("Adjusted Mutual Info Score: " ,metrics.adjusted_mutual_info_score(labels,  clusters))
    return metrics.homogeneity_score(labels, clusters),metrics.completeness_score(labels, clusters), metrics.v_measure_score(labels, clusters),metrics.adjusted_rand_score(labels, clusters),metrics.adjusted_mutual_info_score(labels,  clusters)
