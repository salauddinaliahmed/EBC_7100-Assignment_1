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
    