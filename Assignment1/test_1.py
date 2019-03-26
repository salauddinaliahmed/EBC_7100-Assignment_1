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

    
    
    import nltk
nltk.download('gutenberg')
nltk.download('punkt')
from nltk.corpus import gutenberg 
import random
import os
import glob
import sys
import errno


#files_en = gutenberg.fileids()      # Get file ids
book_1 = gutenberg.open('whitman-leaves.txt').readlines()
book_2 = gutenberg.open('melville-moby_dick.txt').read()
book_3 = gutenberg.open('shakespeare-caesar.txt').read()

sentence_detector = nltk.data.load('tokenizers/punkt/{0}.pickle'.format('english'))
listOfSentences = []
print (type(book_1))

for text in book_1:
    try:
        listOfSentences += [x.replace("\n", " ").replace("  "," ").strip() for x in random.sample(sentence_detector.tokenize(text), 200)]

    except IOError as exc:
    # Do not fail if a directory is found, just ignore it.
        if exc.errno != errno.EISDIR:
            raise

random_sample_input = random.sample(listOfSentences, 15)
print(random_sample_input)

# This block of code writes the result of the previous to a new file
random_sample_output = open("randomsample", "w", encoding='utf-8')
random_sample_input = map(lambda x: x+"\n", random_sample_input)
random_sample_output.writelines(random_sample_input)
random_sample_output.close()

# Final work from Sunday - 26th Jan


import nltk
import random
from nltk.corpus import gutenberg

files_en = gutenberg.fileids()      # Get file ids
moby_dick = gutenberg.raw('melville-moby_dick.txt')
bible = gutenberg.raw('bible-kjv.txt')
parents = gutenberg.raw('edgeworth-parents.txt')
author_list = {'Melville': moby_dick,'KJV': bible, 'Edgeworth':parents}
temp_list = []
with open("complete_book.txt", "w") as f:
    for author,book in author_list.items():
        n = 0  
        book_sent = nltk.sent_tokenize(book)
        print (type(book_sent))
        doc_length = len(book_sent)
    #print (author, "This is the number of senteces in the book", doc_length, book_sent[0])
        parts = int(doc_length/200)
    #print (author, "number of sentences in 1 document", parts)
        for i in book_sent:
            if len(temp_list) <= 200 or n <= doc_length:
                #print (author, "line number, n ", n)
                temp_string = str(book_sent[n:n+parts]) +"|"+author+"|"
                temp_list.append(temp_string)
                n = n+parts+1
            else:
                print (len(temp_list))
                break
        f.write(str(temp_list))
f.close()
same_file = open("complete_book.txt", 'r', encoding='utf-8')
r = same_file.read()
new_list += [x.replace("\n", " ").replace("  "," ").replace("''","") for x in temp_list]
print (new_list[1:20])
