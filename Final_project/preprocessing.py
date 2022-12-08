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
from functions import pca_Plot
from functions import calcPCA

#Preprocessing for clustering - returns a list of cuisines with ingredients
def clustering_preprocessing(json):
    dictCuisineIng = {}
    cuisines = []
    ingredients = []
    
    for i in range(len(json)):   
        cuisine = json[i]['cuisine']
            
        ingredCuisine = json[i]['ingredients']
        
        if cuisine not in dictCuisineIng.keys():
            cuisines.append(cuisine)
            dictCuisineIng[cuisine] = ingredCuisine
            
        else: 
            recpList = dictCuisineIng[cuisine]
            recpList.extend(ingredCuisine)
            dictCuisineIng[cuisine] = recpList
                 
        ingredients.extend(ingredCuisine)
         
    ingredients = list(set(ingredients)) # unique list of ALL ingredients
    numUniqueIngredients = len(ingredients)
    numCuisines = len(cuisines)
    
    vectorizertr = TfidfVectorizer(lowercase= False)
    cluster_vector = vectorizertr.fit_transform(ingredCuisine)   
    
    return cluster_vector, ingredCuisine, numCuisines, numUniqueIngredients, cuisines, ingredients


#Pre-processing for classification
def classify_preprocessing(size):
    
    traindf = pd.read_json("https://raw.githubusercontent.com/salauddinaliahmed/EBC_7100-Assignment_1/master/train.json")
    cuisines  = ['italian',
                 'mexican',
                 'southern_us',
                 'indian',
                 'chinese',
                 'french',
                 'cajun_creole',
                 'thai',
                 'japanese',
                 'greek',
                 'spanish',
                 'korean',
                 'vietnamese',
                 'moroccan',
                 'british',
                 'filipino',
                 'irish',
                 'jamaican',
                 'russian',
                 'brazilian']
    df_sample = pd.DataFrame(columns = ['cuisine','id', 'ingredients'])

    traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  

    cuisine_sample = size

    for cuisine in cuisines:
        df_sample = df_sample.append(traindf[traindf.cuisine == cuisine].head(cuisine_sample), ignore_index=True,sort=True)

    #Word Lemmetization
    traindf['ingredients'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

    #Shuffle the records to avoid bias
    df_sample = shuffle(df_sample)
    df_y = df_sample['cuisine']

    df_sample['ingredients'] = [' , '.join(z).strip() for z in df_sample['ingredients']]  

    def comma_split(s):
        return s.split(',')

    #Preparing TFIDF vector
    vectorizertr = TfidfVectorizer(stop_words='english', tokenizer=comma_split,
                                ngram_range = ( 1 , 1 ),analyzer="word", 
                                max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, lowercase=False)
    
    #Performing TFIDF vectorization on the dataset without labels
    complete_vector_not_reduced = vectorizertr.fit_transform(df_sample['ingredients']).todense()
    
    #For classification - Splitting the data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(complete_vector_not_reduced, df_y, test_size=0.20, random_state = 2)
    
    #Determine how many components in the PCA by plotting variance curve
    pca_Plot(complete_vector_not_reduced)
    
    #Performing PCA by selecting 1000 compoenents and returning the new vector with reduced dataset
    complete_vector =calcPCA(1000, complete_vector_not_reduced)
    
    return complete_vector, x_train, x_test, y_train, y_test, df_sample, df_sample['ingredients']
