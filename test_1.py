import nltk
import random
from nltk.corpus import gutenberg

files_en = gutenberg.fileids()      # Get file ids
moby_dick = gutenberg.open('melville-moby_dick.txt').read()
moby_full = nltk.sent_tokenize(moby_dick)
n = 0
temp_list = []
key_name = moby_dick
for i in range(200):
    temp_string = str(moby_full[n:n+40]) + "|MOBY_DICK|"
    temp_list.append(temp_string)
    n += 41
for each_item in temp_list:
    print (each_item)
    print ("\n")
    
    
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
