from nltk.corpus import gutenberg
import random
import nltk
book_1 = gutenberg.raw('bible-kjv.txt')
book_1 = nltk.sent_tokenize(book_1)
user_list = []
print (len(book_1))
n = 0;
limit = len(book_1)/30
for i in range(int(limit)):
    user_list.append(book_1[n:n+30])
    n+=31
print (len(user_list))
new_sample = random.sample(user_list, 200)
record_string = " ".join(new_sample[1])
record_word_tokenized = nltk.word_tokenize(record_string)
print (len(record_word_tokenized))