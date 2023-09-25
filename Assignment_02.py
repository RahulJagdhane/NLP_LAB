'''
Name : Rahul Kailas jagdhane 
Roll No: 28
Assignment No :02
Title: Bag of Words and TF-IDF using Gensim Library
'''

# importing libraries
import gensim 
from gensim import *
from gensim.utils import simple_preprocess
import numpy as np

# reading content from sample_txt 
text1 = open('As_2sampletext.txt', encoding ='utf-8')
 
tokens1 =[]
for line in text1.read().split('.'):
  tokens1.append(simple_preprocess(line, deacc = True))

g_dict1 = corpora.Dictionary(tokens1)

print("The dictionary has: " +str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)




g_bow =[g_dict1.doc2bow(token, allow_update = True) for token in tokens1]
print("\n\nBag of Words : ", g_bow)
print("\n\n")

g_tfidf = gensim.models.TfidfModel(g_bow, smartirs='ntc')
print("\nTF-IDF Vector:\n")
for item in g_tfidf[g_bow]:
    print([[g_dict1[id], np.around(freq, decimals=2)] for id, freq in item])

