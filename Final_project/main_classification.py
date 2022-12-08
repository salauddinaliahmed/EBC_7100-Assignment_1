#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from preprocessing import classify_preprocessing,clustering_preprocessing
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
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from functions import learning_curve_graph, accuracy_plot, generate_error_eval,plot_confusion_matrix, clean_class, stopwords_gen
import pandas as pd
import json
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import classify_preprocessing
import tensorflow as tf
from tensorflow import keras
import seaborn as sn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping
from urllib.request import urlopen

#number of cuisines
numCuisine = 400

with urlopen('https://raw.githubusercontent.com/salauddinaliahmed/EBC_7100-Assignment_1/master/train.json') as data_file:    
    data = json.loads(data_file.read()) 

#Classification
complete_vector, x_train, x_test, y_train, y_test, df_sample, x_raw= classify_preprocessing(numCuisine)
df_y = df_sample["cuisine"]

cluster_vector, dictCuisineIngred, numCuisines, numIngred, cuisines, ingredients = clustering_preprocessing(data)

stop_words, stop_tuple = stopwords_gen(dictCuisineIngred)

#Cleaning_the_train : removing the stop words
list_x = []
for each_item in x_raw.items():
    new_dummy = []
    new_dummy.append((each_item[1]).split(','))
    list_x.append(new_dummy)
    new_dummy = []

clean_x_train= clean_class(list_x,stop_words)

new_clean_list = []
for each_t in clean_x_train:
    new_clean_list.append(",".join(each_t))


def comma_split(s):
        return s.split(',')
vectorizertr = TfidfVectorizer(tokenizer=comma_split,
                                ngram_range = ( 1 , 1 ),analyzer="word", 
                                max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, lowercase=False)
complete_vector = vectorizertr.fit_transform(new_clean_list)
x_train, x_test = train_test_split(complete_vector, test_size=0.20, random_state = 2)

def comma_split(s):
   return s.split(',')
cuisines  =     ['0:italian',
                 '1:mexican',
                 '2:southern_us',
                 '3:indian',
                 '4:chinese',
                 '5:french',
                 '6:cajun_creole',
                 '7:thai',
                 '8:japanese',
                 '9:greek',
                 '10:spanish',
                 '11:korean',
                 '12:vietnamese',
                 '13:moroccan',
                 '14:british',
                 '15:filipino',
                 '16:irish',
                 '17:jamaican',
                 '18:russian',
                 '19:brazilian']

k_folds = 10

#-------------------- TFIDF-SVM -------------------#
print ("=======================================================================")
print ("Support Vector Machine-----------------------------------------------------------------------")
print ("=======================================================================")
clf_svm = svm.SVC(C=1.0,
                  kernel='linear',
                  degree=3, 
                  gamma='auto_deprecated', 
                  coef0=0.0, 
                  shrinking=True,
                  probability=False, 
                  tol=0.001, 
                  cache_size=2000, 
                  class_weight=None,
                  verbose=False, 
                  max_iter=-1, 
                  decision_function_shape='ovr',
                  random_state=None)

clf_svm.fit(x_train, y_train)
print ("SVM Classifier accuracy for TFIDF: ", "{:.3%}".format(clf_svm.score(x_test,y_test)))

#Performing cross validation for SVM TFIDF
svm_tfidf_scores = cross_val_score(estimator=clf_svm, 
                        X=complete_vector, 
                        y=df_y, 
                        cv=k_folds)

print('cross-validation accuracy scores TFIDF SVM: %s' % svm_tfidf_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(svm_tfidf_scores), np.std(svm_tfidf_scores)))
#print ("Validation Curve for SVM TFIDF")
#validation_curve_graph(complete_vector, df_y, clf_svm, "gamma","Validation Curve with SVM")
print ("Learning Curve for SVM TFIDF")
learning_curve_graph(clf_svm, complete_vector, df_y)
print("Accuracy Plot fot SVM TFIDF")
accuracy_plot(k_folds, svm_tfidf_scores, clf_svm,"SVM")

#ERROR EVALUATION
print ("------------Error Evaluation for SVM-------------")
print ("Error Evaluation for SVM TFIDF")
svm_tfidf_confu_mat = generate_error_eval(clf_svm, complete_vector, df_y, cuisines, k_folds)

print("Graphs - SVM TFIDF")
plt.figure()
plot_confusion_matrix(svm_tfidf_confu_mat, classes= cuisines,
                     title='Confusion matrix, without normalization')
#Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(svm_tfidf_confu_mat, classes=cuisines, normalize=True,
                     title='Normalized confusion matrix')
plt.show()

#-------------------- TFIDF-KNN -------------------#
print ("=======================================================================")
print ("k-nearest neighbors--------------------------------------------------------------------")
print ("=======================================================================")
clf_knn = KNeighborsClassifier(weights='uniform', 
                               algorithm='auto', 
                               leaf_size=30, 
                               p=2, 
                               metric_params=None, 
                               n_jobs=1,
                               n_neighbors=10)

clf_knn.fit(x_train, y_train)
print ("KNN Classifier accuracy for TFIDF: ", "{:.3%}".format(clf_knn.score(x_test,y_test)))

#Performing cross validation for KNN TFIDF
knn_tfidf_scores = cross_val_score(estimator=clf_knn, 
                        X=complete_vector, 
                        y=df_y, 
                        cv=k_folds)

print('cross-validation accuracy scores TFIDF KNN: %s' % knn_tfidf_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(knn_tfidf_scores), np.std(knn_tfidf_scores)))
#print ("Validation Curve for KNN TFIDF")
#validation_curve_graph(complete_vector, df_y, clf_knn,"n_neighbors","Validation Curve with KNN")
print ("Learning Curve for KNN TFIDF")
learning_curve_graph(clf_knn, complete_vector, df_y)
print("Accuracy Plot fot KNN TFIDF")
accuracy_plot(k_folds, knn_tfidf_scores, clf_knn,"KNN")

#ERROR EVALUATION
print ("------------Error Evaluation for KNN-------------")
print ("Error Evaluation for KNN TFIDF")
knn_tfidf_confu_mat = generate_error_eval(clf_knn, complete_vector, df_y,cuisines, k_folds)

print("Graphs - KNN TFIDF")
plt.figure()
plot_confusion_matrix(knn_tfidf_confu_mat, classes= cuisines,
                     title='Confusion matrix, without normalization')
#Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(knn_tfidf_confu_mat, classes=cuisines, normalize=True,
                     title='Normalized confusion matrix')
plt.show()
#-------------------- TFIDF-Decision Tree -------------------#
print ("=======================================================================")
print ("Decision Tree--------------------------------------------------------------------")
print ("=======================================================================")
clf_dtree = tree.DecisionTreeClassifier(criterion='gini', 
                                        splitter='best', 
                                        max_depth=None, 
                                        min_samples_split=2, 
                                        min_samples_leaf=1, 
                                        min_weight_fraction_leaf=0.0, 
                                        max_features=None, 
                                        random_state=None, 
                                        max_leaf_nodes=None, 
                                        min_impurity_decrease=0.0, 
                                        min_impurity_split=None, 
                                        class_weight=None, 
                                        presort=False)

clf_dtree.fit(x_train, y_train)
#Performing cross validation for Decision Tree TFIDF
dtree_tfidf_scores = cross_val_score(estimator=clf_dtree, 
                        X=complete_vector, 
                        y=df_y, 
                        cv=k_folds)
print ("Decision Tree Classifier accuracy for TFIDF: ", "{:.3%}".format(clf_dtree.score(x_test,y_test)))
print('cross-validation accuracy scores TFIDF Decision Tree: %s' % dtree_tfidf_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(dtree_tfidf_scores), np.std(dtree_tfidf_scores)))
#print ("Validation Curve for Decision Tree TFIDF")
#validation_curve_graph(complete_vector, df_y, clf_dtree,"max_depth","Validation Curve with Decision Tree")
print ("Learning Curve for Decision Tree TFIDF")
learning_curve_graph(clf_dtree, complete_vector, df_y)
print("Accuracy Plot fot Decision Tree TFIDF")
accuracy_plot(k_folds, dtree_tfidf_scores, clf_dtree,"Decision Tree")

#ERROR EVALUATION
print ("------------Error Evaluation for Decision Tree-------------")
print ("Error Evaluation for Decision Tree TFIDF")
dtree_tfidf_confu_mat = generate_error_eval(clf_dtree, complete_vector, df_y,cuisines, k_folds)

print("Graphs - Decision Tree TFIDF")
plt.figure()
plot_confusion_matrix(dtree_tfidf_confu_mat, classes= cuisines,
                     title='Confusion matrix, without normalization')
#Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(dtree_tfidf_confu_mat, classes=cuisines, normalize=True,
                     title='Normalized confusion matrix')
plt.show()

#-------------------- TFIDF-Random Forest -------------------#
print ("=======================================================================")
print ("Random Forest--------------------------------------------------------------------")
print ("=======================================================================")
clf_rf = RandomForestClassifier(bootstrap=True,
                                class_weight=None,
                                criterion='gini',
                                max_depth=2, 
                                max_features='auto',
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                min_samples_leaf=1,
                                min_samples_split=2,
                                min_weight_fraction_leaf=0.0, 
                                n_estimators=100, 
                                n_jobs=None,
                                oob_score=False, 
                                random_state=0, 
                                verbose=0, 
                                warm_start=False)
clf_rf.fit(x_train, y_train)
#Performing cross validation for Random Forest TFIDF
rf_tfidf_scores = cross_val_score(estimator=clf_rf, 
                        X=complete_vector, 
                        y=df_y, 
                        cv=k_folds)
print ("Random Forest Classifier accuracy for TFIDF: ", "{:.3%}".format(clf_rf.score(x_test,y_test)))
print('cross-validation accuracy scores TFIDF Random Forest: %s' % rf_tfidf_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(rf_tfidf_scores), np.std(rf_tfidf_scores)))
print ("Learning Curve for Random Forest TFIDF")
learning_curve_graph(clf_rf, complete_vector, df_y)
print("Accuracy Plot fot Random Forest TFIDF")
accuracy_plot(k_folds, rf_tfidf_scores, clf_rf,"Random Forest")

#ERROR EVALUATION
print ("------------Error Evaluation for Random Forest-------------")
print ("Error Evaluation for Random Forest TFIDF")
rf_tfidf_confu_mat = generate_error_eval(clf_rf, complete_vector, df_y,cuisines, k_folds)

print("Graphs - Random Forest TFIDF")
plt.figure()
plot_confusion_matrix(rf_tfidf_confu_mat, classes= cuisines,
                     title='Confusion matrix, without normalization')
#Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(rf_tfidf_confu_mat, classes=cuisines, normalize=True,
                     title='Normalized confusion matrix')
plt.show()

##Taking user input here for prediction
user_input = input("Please enter a comma seperated ingredients: ")
myList = user_input.split(",")

#Removing the stop words from the given list of ingredients
clean_x_train_1= clean_class(list_x,stop_words)

clean_x_train_1.append(myList)

new_clean_list_1 = []
for each_t in clean_x_train:
    new_clean_list_1.append(",".join(each_t))

#Performing the TFIDF vector on the given list of ingredients
def comma_split(s):
        return s.split(',')
vectorizertr = TfidfVectorizer(tokenizer=comma_split,
                                ngram_range = ( 1 , 1 ),analyzer="word", 
                                max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, lowercase=False)
complete_vector_1 = vectorizertr.fit_transform(new_clean_list_1)

#Predicting the cusine based on the given set of ingredients
outputClass = clf_svm.predict(complete_vector_1[-1])
print ("This is the prediction value",outputClass)