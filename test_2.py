# Working code for collection chucks from each book. 
# We now have each chunck containing 30 sentences from each book. This will give us enough chunks to sample 200 from each
from nltk.corpus import gutenberg, stopwords
import random
import nltk
book_1 = gutenberg.raw('bible-kjv.txt')
book_2 = gutenberg.raw('melville-moby_dick.txt')
book_3 = gutenberg.raw('edgeworth-parents.txt')
dict_labels = {'KJV (Bible)': book_1, 'Herman Melville (Moby-Dick)' : book_2, 'Richard Lovell Edgeworth (Parents)' : book_3}
master_list = []
# Dividing each book into documents of 30 sentences. appending them to *master_list*. 
for author, book in dict_labels.items():
    n = 0;
    book = nltk.sent_tokenize(book)
    limit = int(len(book)/31)
    print (author, " number of 30 chunks is:" , limit)
    for i in range(limit):
        master_list.append((book[n:n+30], author))
        n+=31
print (size_of_books)
complete_book_1 = master_list[0:size_of_books[0]-1]
complete_book_2 = master_list[961:1278]       
complete_book_3 = master_list[1278:-1]
total_sample = [complete_book_1, complete_book_2, complete_book_3]
for i in total_sample:
    print ("Total number of documents in each book are", len(i))

#Random Sampling 200 documents from each book. 

sample_book1 = random.sample(complete_book_1, 200)
sample_book2 = random.sample(complete_book_2, 200)
sample_book3 = random.sample(complete_book_3, 200)

#Preprocessing the data, by taking 150 top frequency words excluding the stop words from each document. 
book1_words = [nltk.word_tokenize(" ".join(each_doc).lower()) for each_doc in sample_book1]
book2_words = [nltk.word_tokenize(" ".join(each_doc).lower()) for each_doc in sample_book2]
book3_words = [nltk.word_tokenize(" ".join(each_doc).lower()) for each_doc in sample_book3]

#initializing stopwords
stop_words = stopwords.words("english")

cleaned_book1 = book1_words
cleaned_book2 = book2_words
cleaned_book3 = book3_words
iterator_for_clean_books = [cleaned_book1, cleaned_book2, cleaned_book3]
for each_book in iterator_for_clean_books:
    for each_list in each_book:
        for each_item in each_list:
            if each_item in stop_words:
                each_list.remove(each_item)
print (len(cleaned_book1[0]))
print ("\n")
print (len(cleaned_book2[0]))
print ("\n")
print ((cleaned_book3[0]))
