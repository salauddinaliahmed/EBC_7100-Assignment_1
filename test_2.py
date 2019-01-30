# Working code for collection chucks from each book. 
# We now have each chunck containing 30 sentences from each book. This will give us enough chunks to sample 200 from each book.
from nltk.corpus import gutenberg
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


        

