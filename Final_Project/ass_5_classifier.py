#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:39:04 2019

@author: salauddinali
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn import svm, tree, metrics
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

traindf = pd.read_json("/Users/salauddinali/Downloads/whats-cooking/train.json")

cuisines  = ['italian','mexican','southern_us','indian','chinese','french','cajun_creole','thai','japanese','greek','spanish','korean','vietnamese','moroccan','british','filipino','irish','jamaican','russian','brazilian']
df_sample = pd.DataFrame(columns = ['cuisine','id', 'ingredients'])
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  

cuisine_sample = 150

for cuisine in cuisines:
    df_sample = df_sample.append(traindf[traindf.cuisine == cuisine].head(cuisine_sample), ignore_index=True)

#traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

df_sample = shuffle(df_sample)

df_y = df_sample['cuisine']

df_sample['ingredients'] = [' , '.join(z).strip() for z in df_sample['ingredients']]  

#testdf = pd.read_json("/Users/salauddinali/Downloads/whats-cooking/train.json") 
#testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
#testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

def comma_split(s):
   return s.split(',')


clf_svm = svm.SVC(kernel="linear")

corpustr = df_sample['ingredients'] 
vectorizertr = TfidfVectorizer(stop_words='english', tokenizer=comma_split,
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, lowercase=False)
tfidftr=vectorizertr.fit_transform(corpustr).todense()

x_train, x_test, y_train, y_test = train_test_split(tfidftr, df_y, test_size=0.20, random_state = 2)


corpusts = x_test

"""testdf['ingredients_clean_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts)
"""
predictors_tr = x_train

targets_tr = y_train

predictors_ts = x_test


#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
#parameters = {'C':[1, 10]}
#clf = LinearSVC()
#clf = LogisticRegression(multi_class="auto")

#classifier = GridSearchCV(clf, parameters)

#classifier=classifier.fit(predictors_tr,targets_tr)

#predictions=classifier.predict(predictors_ts)
#testdf['cuisine'] = predictions
clf_dtree = tree.DecisionTreeClassifier()
clf_knn = KNeighborsClassifier(n_neighbors=8)


clf_svm.fit(x_train, y_train)
print ("SVM Classifier accuracy for TFIDF: ", "{:.3%}".format(clf_svm.score(x_test,y_test)))

clf_dtree = clf_dtree.fit(x_train, y_train)
print ("Decision Tree accuracy for BOW: ","{:.3%}".format(clf_dtree.score(x_test, y_test)))
#Knn for BOW
clf_knn.fit(x_train, y_train) 
print("KNN accuracy for BOW: ","{:.3%}".format(clf_knn.score(x_test, y_test)))