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


'''
OUTPUT::

The dictionary has: 63 tokens

{'abdul': 0, 'beginning': 1, 'birth': 2, 'book': 3, 'class': 4, 'early': 5, 'family': 6, 'focuses': 7, 'his': 8, 'in': 9, 'india': 10, 'into': 11, 'kalam': 12, 'life': 13, 'middle': 14, 'on': 15, 'rameswaram': 16, 'tamil': 17, 'the': 18, 'with': 19, 'boat': 20, 'community': 21, 'considered': 22, 'father': 23, 'of': 24, 'owned': 25, 'sign': 26, 'wealth': 27, 'an': 28, 'close': 29, 'each': 30, 'everyone': 31, 'helped': 32, 'ideal': 33, 'it': 34, 'knit': 35, 'making': 36, 'other': 37, 'place': 38, 'raise': 39, 'society': 40, 'to': 41, 'was': 42, 'where': 43, 'age': 44, 'all': 45, 'and': 46, 'discuss': 47, 'encouraged': 48, 'from': 49, 'openly': 50, 'religions': 51, 'respect': 52, 'spirituality': 53, 'taught': 54, 'young': 55, 'as': 56, 'at': 57, 'imam': 58, 'mosque': 59, 'neighborhood': 60, 'served': 61, 'well': 62}


Bag of Words :  [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1)], [(8, 1), (9, 1), (18, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1)], [(6, 1), (16, 1), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1), (35, 1), (36, 1), (37, 1), (38, 1), (39, 1), (40, 1), (41, 1), (42, 1), (43, 1)], [(12, 1), (41, 2), (42, 1), (44, 1), (45, 1), (46, 1), (47, 1), (48, 1), (49, 1), (50, 1), (51, 1), (52, 1), (53, 1), (54, 1), (55, 1)], [(8, 1), (18, 1), (23, 1), (28, 1), (56, 2), (57, 1), (58, 1), (59, 1), (60, 1), (61, 1), (62, 1)], []]




TF-IDF Vector:

[['abdul', 0.25], ['beginning', 0.25], ['birth', 0.25], ['book', 0.25], ['class', 0.25], ['early', 0.25], ['family', 0.16], ['focuses', 0.25], ['his', 0.11], ['in', 0.16], ['india', 0.25], ['into', 0.25], ['kalam', 0.16], ['life', 0.25], ['middle', 0.25], ['on', 0.25], ['rameswaram', 0.16], ['tamil', 0.25], ['the', 0.11], ['with', 0.25]]
[['his', 0.15], ['in', 0.22], ['the', 0.15], ['boat', 0.35], ['community', 0.35], ['considered', 0.35], ['father', 0.22], ['of', 0.35], ['owned', 0.35], ['sign', 0.35], ['wealth', 0.35]]
[['family', 0.17], ['rameswaram', 0.17], ['an', 0.17], ['close', 0.26], ['each', 0.26], ['everyone', 0.26], ['helped', 0.26], ['ideal', 0.26], ['it', 0.26], ['knit', 0.26], ['making', 0.26], ['other', 0.26], ['place', 0.26], ['raise', 0.26], ['society', 0.26], ['to', 0.17], ['was', 0.17], ['where', 0.26]]
[['kalam', 0.17], ['to', 0.34], ['was', 0.17], ['age', 0.26], ['all', 0.26], ['and', 0.26], ['discuss', 0.26], ['encouraged', 0.26], ['from', 0.26], ['openly', 0.26], ['religions', 0.26], ['respect', 0.26], ['spirituality', 0.26], ['taught', 0.26], ['young', 0.26]]
[['his', 0.13], ['the', 0.13], ['father', 0.19], ['an', 0.19], ['as', 0.6], ['at', 0.3], ['imam', 0.3], ['mosque', 0.3], ['neighborhood', 0.3], ['served', 0.3], ['well', 0.3]]



'''