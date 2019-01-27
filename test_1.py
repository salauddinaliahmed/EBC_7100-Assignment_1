import nltk
import random
from nltk.corpus import gutenberg

files_en = gutenberg.fileids()      # Get file ids
moby_dick = gutenberg.raw('melville-moby_dick.txt')
bible = gutenberg.raw('bible-kjv.txt')
parents = gutenberg.raw('edgeworth-parents.txt')
author_list = {'Melville': moby_dick,'KJV': bible, 'Edgeworth':parents}
temp_list = []
for author,book in author_list.items():
    n = 0  
    book_sent = nltk.sent_tokenize(book)
    doc_length = len(book_sent)
    #print (author, "This is the number of senteces in the book", doc_length, book_sent[0])
    parts = int(doc_length/200)
    print (author, "number of sentences in 1 document", parts)
    for i in book_sent:
        if len(temp_list) < 200 or n < doc_length:
            temp_string = str(book_sent[n:n+parts]) +"|"+author+"|"
            temp_list.append(temp_string)
            n = n+parts+1
        else:
            break
print (temp_list[598])