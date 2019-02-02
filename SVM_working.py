from nltk.corpus import gutenberg
import random
import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn import svm

book_1 = gutenberg.raw('bible-kjv.txt').lower()
book_2 = gutenberg.raw('melville-moby_dick.txt').lower()
book_3 = gutenberg.raw('edgeworth-parents.txt').lower()
dict_labels = {'KJV (Bible)': book_1, 'Herman Melville (Moby-Dick)' : book_2, 'Richard Lovell Edgeworth (Parents)' : book_3}
master_list = []
size_of_books = []
# Dividing each book into documents of 30 sentences. appending them to *master_list*. 
for author, book in dict_labels.items():
    n = 0;
    book = nltk.sent_tokenize(book)
    limit = int(len(book)/31)
    size_of_books.append(limit)
    for i in range(limit):
        master_list.append(" ".join(book[n:n+30]))
        n+=31 
        
#Seperating the 3 books.
complete_book_1 = (master_list[0:size_of_books[0]-1])
complete_book_2 = (master_list[961:1278])      
complete_book_3 = (master_list[1278:-1])

#Sampling 200 documents from the entire book.
sample_book1 = pd.DataFrame(random.sample(complete_book_1, 200))
sample_book2 = pd.DataFrame(random.sample(complete_book_2, 200))
sample_book3 = pd.DataFrame(random.sample(complete_book_3, 200))

#Converting Labels to integers
kjv_bible = 0
melville_moby_dick = 1
edgeworth_parents = 2

# Adding a new column to the sample books containing lables. 
sample_book1['Label'] = kjv_bible
sample_book2['Label'] = melville_moby_dick
sample_book3['Label'] = edgeworth_parents

frames = [sample_book1, sample_book2, sample_book3]
complete_sample = pd.concat(frames)
complete_sample = shuffle(complete_sample)

x_train, x_test, y_train, y_test = train_test_split(complete_sample[0], complete_sample['Label'], test_size=0.2, random_state=4)


#BOW 
token = RegexpTokenizer(r'[a-zA-Z]+')
cv_bow = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1), analyzer='word', tokenizer = token.tokenize)
filtered_counts_bow= cv_bow.fit_transform(x_train)
fil_test = cv_bow.fit_transform(x_test)
bow = filtered_counts_bow.toarray()

clf = svm.SVC(kernel = "linear")
clf.fit(filtered_counts_bow, y_train)


#TFIDF
token = RegexpTokenizer(r'[a-zA-Z]+')
cv_tfidf = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1), analyzer='word', tokenizer = token.tokenize)
filtered_counts_tfidf= cv_tfidf.fit_transform(x_train)
tfidf = filtered_counts_tfidf.toarray()
