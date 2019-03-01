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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

lemm = WordNetLemmatizer()
def preprocess(topwordsnum):

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
        
    def most_common_words():
        most_common_book1 = []
        for each_doc in clean_book1_words:
            a = Counter(each_doc).most_common(topwordsnum)
            most_common_book1.append(" ".join(i[0] for i in a))
        most_common_book2 = []
        for each_doc in clean_book2_words:
            a = Counter(each_doc).most_common(topwordsnum)
            most_common_book2.append(" ".join(i[0] for i in a))
        most_common_book3 = []
        for each_doc in clean_book3_words:
            a = Counter(each_doc).most_common(topwordsnum)
            most_common_book3.append(" ".join(i[0] for i in a))
        return most_common_book1, most_common_book2, most_common_book3
   
    
 #### Collocations - bigrams. 
   
    def most_common_bigrams():
        most_common_book1 = []
        for each_doc in clean_book1_words:
            to_string_conv = " ".join(each_doc)
            to_string_conv = nltk.word_tokenize(to_string_conv)
            a = BigramCollocationFinder.from_words(to_string_conv)
            each_doc_colloc = list(a.nbest(BigramAssocMeasures.pmi, 75))
            most_common_book1.append(" ".join(each_word for each_titem in each_doc_colloc for each_word in each_titem))
        most_common_book2 = []
        for each_doc in clean_book2_words:
            to_string_conv = " ".join(each_doc)
            to_string_conv = nltk.word_tokenize(to_string_conv)
            a = BigramCollocationFinder.from_words(to_string_conv)
            each_doc_colloc = list(a.nbest(BigramAssocMeasures.pmi, 75))
            most_common_book2.append(" ".join(each_word for each_titem in each_doc_colloc for each_word in each_titem))
        most_common_book3 = []
        for each_doc in clean_book3_words:
            to_string_conv = " ".join(each_doc)
            to_string_conv = nltk.word_tokenize(to_string_conv)
            a = BigramCollocationFinder.from_words(to_string_conv)
            each_doc_colloc = list(a.nbest(BigramAssocMeasures.pmi, 75))
            most_common_book3.append(" ".join(each_word for each_titem in each_doc_colloc for each_word in each_titem))
        return most_common_book1, most_common_book2, most_common_book3
    
    if isinstance(topwordsnum, int):
        most_common_book1, most_common_book2, most_common_book3 = most_common_words()
    else:
        most_common_book1, most_common_book2, most_common_book3 = most_common_bigrams()
    
    
    #Once the samples are ready, we are converting it into dataframe for training the model.
    
    #print (most_common_book2)

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
    
    return bow_vector, tfidf_vector, df_x, df_y

