# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:22:25 2019

@author: Vrushti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 18:37:15 2019
EBC 7100 Assignment 1 - Text Classification using NLTK with Python for supervised learning

author Jiaxi Chen (300011851), Salauddin Ali Ahmed (300031318), Vrushti Buch (8844799)

"""


#Importing required libraries. We have used NLTK, ScikitLearn and basic python libraries
#to perform supervised machine learning.

import nltk
import pandas as pd
from nltk.corpus import gutenberg
import random
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from itertools import cycle

#Following libraries are used to supress warnings, the details can be found at
#: https://blog.csdn.net/Homewm/article/details/84524558 
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


# Dividing each book into documents of 30 sentences. appending them to *master_list*. 
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

#Selecting top 150 most common words.
from collections import Counter
most_common_book1 = []
for each_doc in clean_book1_words:
    a = Counter(each_doc[0]).most_common(150)
    most_common_book1.append(" ".join(each_doc))

most_common_book2 = []
for each_doc in clean_book2_words:
    a = Counter(each_doc[0]).most_common(150)
    most_common_book2.append(" ".join(each_doc))

most_common_book3 = []
for each_doc in clean_book3_words:
    a = Counter(each_doc[0]).most_common(150)
    most_common_book3.append(" ".join(each_doc))

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

#Seperating Labels and Data from the complete sample.
df_x = (complete_sample['Data'])
df_y = (complete_sample['Label'])

#BOW Vectorization on the complete sample.
cv_bow = CountVectorizer(lowercase=False)
bow_vector= cv_bow.fit_transform(df_x)

#TFIDF Vectorization on the complete sample.
tfidf = TfidfVectorizer(lowercase=False)
tfidf_vector = tfidf.fit_transform(df_x)

#Split the matrix from BOW vectorizer into test and train.
x_train, x_test, y_train, y_test = train_test_split(bow_vector, df_y, test_size=0.20, random_state = 4)

#Classifiers
clf_svm = svm.SVC(kernel="linear",probability=True)
clf_dtree = tree.DecisionTreeClassifier()
clf_knn = KNeighborsClassifier(n_neighbors=8)


#SVM Classifer on BOW
svm_bow_model = clf_svm.fit(x_train, y_train)
print ("SVM Classifier:","{:.3%}".format(clf_svm.score(x_test, y_test)))

#Decision Tree on BOW
clf_dtree = clf_dtree.fit(x_train, y_train)
print ("Decision Tree: ","{:.3%}".format(clf_dtree.score(x_test, y_test)))

#Knn for BOW
clf_knn.fit(x_train, y_train) 
print("KNN: ","{:.3%}".format(clf_knn.score(x_test, y_test)))

#Split the matrix from TFIDF vectorizer into test and train.
x_train, x_test, y_train, y_test = train_test_split(tfidf_vector, df_y, test_size=0.2, random_state = 4)

print ("===============================")
#SVM for TFIDF
clf_svm.fit(x_train, y_train,)
print ("SVM Classifier accuracy: ", "{:.3%}".format(clf_svm.score(x_test,y_test)))

#Decision Tree for TFIDF
clf_dtree = clf_dtree.fit(x_train, y_train)
print ("Decision Tree accuracy: ","{:.3%}".format(clf_dtree.score(x_test,y_test)))

#Knn for TFIDF
clf_knn.fit(x_train, y_train) 
print("KNN accuracy: ", "{:.3%}".format(clf_knn.score(x_test, y_test)))

#Ten-fold cross-validation
k_folds = 10

#Performing cross validation for SVM TFIDF
svm_tfidf_scores = cross_val_score(estimator=clf_svm, 
                        X=tfidf_vector, 
                        y=df_y, 
                        cv=k_folds)

#Performing cross validation for DTree TFIDF
dtree_tfidf_scores = cross_val_score(estimator=clf_dtree, 
                        X=tfidf_vector, 
                        y=df_y, 
                        cv=k_folds)

#Performing cross validation for KNN TFIDF
knn_tfidf_scores = cross_val_score(estimator=clf_knn, 
                        X=tfidf_vector, 
                        y=df_y, 
                        cv=k_folds)

#Performing cross validation for SVM BOW
svm_bow_scores = cross_val_score(estimator=clf_svm, 
                        X=bow_vector, 
                        y=df_y, 
                        cv=k_folds)

#Performing cross validation for DTree BOW
dtree_bow_scores = cross_val_score(estimator=clf_dtree, 
                        X=bow_vector, 
                        y=df_y, 
                        cv=k_folds)

#Performing cross validation for KNN BOW
knn_bow_scores = cross_val_score(estimator=clf_knn, 
                        X=bow_vector, 
                        y=df_y, 
                        cv=k_folds)


#Function to plot accuracies for each vector
import matplotlib.pyplot as plt
def accuracy_plot(bow_score, tfidf_score, classifier_used):    
    plt.plot(list(range(1,11)),list(bow_score),'b--', list(tfidf_score), 'r--')
    plt.axis([1, k_folds, 0.90, 1.10])
    plt.legend(("BOW", "TFIDF"))
    plt.title(classifier_used)
    plt.xlabel('K folds')
    plt.ylabel('Accuracy')
    plt.show()

print('cross-validation accuracy scores TFIDF SVM: %s' % svm_tfidf_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(svm_tfidf_scores), np.std(svm_tfidf_scores)))
print ("===============================")
print('cross-validation accuracy scores BOW SVM: %s' % svm_bow_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(svm_bow_scores), np.std(svm_bow_scores)))
accuracy_plot(svm_bow_scores, svm_tfidf_scores, "Support Vector Machine - SVM")
print ("===============================")


print('cross-validation accuracy scores TFIDF Dtree: %s' % dtree_tfidf_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(dtree_tfidf_scores), np.std(dtree_tfidf_scores)))
print ("===============================")
print('cross-validation accuracy scores BOW Dtree: %s' % dtree_bow_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(dtree_bow_scores), np.std(dtree_bow_scores)))
accuracy_plot(dtree_bow_scores, dtree_tfidf_scores, "Decision Tree")
print ("===============================")


print('cross-validation accuracy scores TFIDF KNN: %s' % knn_tfidf_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(knn_tfidf_scores), np.std(knn_tfidf_scores)))
print ("===============================")
print('cross-validation accuracy scores BOW KNN: %s' % knn_bow_scores)
print('cross-validation accuracy: %.3f +/- %.3f' % (np.mean(knn_bow_scores), np.std(knn_bow_scores)))
accuracy_plot(knn_bow_scores, knn_tfidf_scores, "KNN")
print ("===============================")


#Error evaluation with classification report and confusion matrix.
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

target_names = ['0: KJV-Bible', '1: Moby-Dick', '2: Edgeworth']

#Creating a function to generate confusion matrix and classification report
#classifier: Any classifier
#train_vector: TFIDF / BOW
#target_variables: Three classes defined abiove
#kfold: Cross validation
#Returns a confusion matrix
def generate_error_eval(classfier, train_vector, target_variables, kfolds):
   error_evaluation = cross_val_predict(estimator=classfier, 
                        X=train_vector, 
                        y=target_variables, 
                        cv=kfolds)
   confu_mat = confusion_matrix(target_variables, error_evaluation)
   print(classification_report(target_variables, error_evaluation, target_names=target_names))
   return confu_mat

print ("Error Evaluation for SVM TFIDF")
svm_tfidf_confu_mat = generate_error_eval(clf_svm, tfidf_vector, df_y, k_folds)

print ("Error Evaluation for SVM BOW")
svm_bow_confu_mat = generate_error_eval(clf_svm, bow_vector, df_y, k_folds)

print ("Error Evaluation for DTREE TFIDF")
dtree_tfidf_confu_mat = generate_error_eval(clf_dtree, tfidf_vector, df_y, k_folds)

print ("Error Evaluation for DTREE BOW")
dtree_bow_confu_mat = generate_error_eval(clf_dtree, bow_vector, df_y, k_folds)

print ("Error Evaluation for KNN TFIDF")
knn_tfidf_confu_mat = generate_error_eval(clf_knn, tfidf_vector, df_y, k_folds)

print ("Error Evaluation for KNN BOW")
knn_bow_confu_mat = generate_error_eval(clf_knn, bow_vector, df_y, k_folds)



#Graphs for confusion matrix.
import itertools
import matplotlib.pyplot as plt

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

  # print(cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.tight_layout()

#### SVM TFIDF ###
# Plot non-normalized confusion matrix

print("Graphs - SVM TFIDF")
plt.figure()
plot_confusion_matrix(svm_tfidf_confu_mat, classes=target_names,
                     title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(svm_tfidf_confu_mat, classes=target_names, normalize=True,
                     title='Normalized confusion matrix')

plt.show()

#### SVM BOW ###

print("Graphs - SVM BOW")

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(svm_bow_confu_mat, classes=target_names,
                     title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(svm_bow_confu_mat, classes=target_names, normalize=True,
                     title='Normalized confusion matrix')

plt.show()

#### DTREE TFIDF ###

print("Graphs - DTREE TFIDF")

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(dtree_tfidf_confu_mat, classes=target_names,
                     title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(dtree_tfidf_confu_mat, classes=target_names, normalize=True,
                     title='Normalized confusion matrix')

plt.show()

#### DTREE BOW ###
print("Graphs - DTREE BOW")

plt.figure()
plot_confusion_matrix(dtree_bow_confu_mat, classes=target_names,
                     title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(dtree_bow_confu_mat, classes=target_names, normalize=True,
                     title='Normalized confusion matrix')

plt.show()
