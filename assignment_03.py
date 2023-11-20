'''Name: Rahul Jagdhane
Batch:B2
Roll no:28
Pract no 3: Generating the n gram model using nltk'''
from nltk import ngrams

from nltk.util import ngrams
#unigram model
print("------Unigram Model------")
n = 1
sentence = 'The story follows the adventures of Monkey D. Luffy and his crew, the Straw Hat Pirates, where he explores the Grand Line in search of the mythical treasure known as the "One Piece" in order to become the next King of the Pirates.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)
print("\n\n")
#bigram model
print("------Bigram Model------")
n = 2
sentence = 'The story follows the adventures of Monkey D. Luffy and his crew, the Straw Hat Pirates, where he explores the Grand Line in search of the mythical treasure known as the "One Piece" in order to become the next King of the Pirates.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)
print("\n\n")
print("------Trigram Model------")
#trigram model
n = 3
sentence = 'The story follows the adventures of Monkey D. Luffy and his crew, the Straw Hat Pirates, where he explores the Grand Line in search of the mythical treasure known as the "One Piece" in order to become the next King of the Pirates.'
unigrams = ngrams(sentence.split(), n)

for item in unigrams:
    print(item)

print("\n\n")
#using text file input
print("------Trigram Model using text file------")
from nltk import ngrams
file = open("sampletext_2.txt")
for i in file.readlines():
    cumulative = i
    sentences = i.split(".")
    counter = 0
    for sentence in sentences:
        trigrams = ngrams(sentence.split(" "), 3)
        print("For sentence", counter + 1, ", trigrams are: ")
        for grams in trigrams:
            print(grams)
        counter += 1
        print()
    
 #output    
'''
------Unigram Model------
('The',)
('story',)
('follows',)
('the',)
('adventures',)
('of',)
('Monkey',)
('D.',)
('Luffy',)
('and',)
('his',)
('crew,',)
('the',)
('Straw',)
('Hat',)
('Pirates,',)
('where',)
('he',)
('explores',)
('the',)
('Grand',)
('Line',)
('in',)
('search',)
('of',)
('the',)
('mythical',)
('treasure',)
('known',)
('as',)
('the',)
('"One',)
('Piece"',)
('in',)
('order',)
('to',)
('become',)
('the',)
('next',)
('King',)
('of',)
('the',)
('Pirates.',)



------Bigram Model------
('The', 'story')
('story', 'follows')
('follows', 'the')
('the', 'adventures')
('adventures', 'of')
('of', 'Monkey')
('Monkey', 'D.')
('D.', 'Luffy')
('Luffy', 'and')
('and', 'his')
('his', 'crew,')
('crew,', 'the')
('the', 'Straw')
('Straw', 'Hat')
('Hat', 'Pirates,')
('Pirates,', 'where')
('where', 'he')
('he', 'explores')
('explores', 'the')
('the', 'Grand')
('Grand', 'Line')
('Line', 'in')
('in', 'search')
('search', 'of')
('of', 'the')
('the', 'mythical')
('mythical', 'treasure')
('treasure', 'known')
('known', 'as')
('as', 'the')
('the', '"One')
('"One', 'Piece"')
('Piece"', 'in')
('in', 'order')
('order', 'to')
('to', 'become')
('become', 'the')
('the', 'next')
('next', 'King')
('King', 'of')
('of', 'the')
('the', 'Pirates.')



------Trigram Model------
('The', 'story', 'follows')
('story', 'follows', 'the')
('follows', 'the', 'adventures')
('the', 'adventures', 'of')
('adventures', 'of', 'Monkey')
('of', 'Monkey', 'D.')
('Monkey', 'D.', 'Luffy')
('D.', 'Luffy', 'and')
('Luffy', 'and', 'his')
('and', 'his', 'crew,')
('his', 'crew,', 'the')
('crew,', 'the', 'Straw')
('the', 'Straw', 'Hat')
('Straw', 'Hat', 'Pirates,')
('Hat', 'Pirates,', 'where')
('Pirates,', 'where', 'he')
('where', 'he', 'explores')
('he', 'explores', 'the')
('explores', 'the', 'Grand')
('the', 'Grand', 'Line')
('Grand', 'Line', 'in')
('Line', 'in', 'search')
('in', 'search', 'of')
('search', 'of', 'the')
('of', 'the', 'mythical')
('the', 'mythical', 'treasure')
('mythical', 'treasure', 'known')
('treasure', 'known', 'as')
('known', 'as', 'the')
('as', 'the', '"One')
('the', '"One', 'Piece"')
('"One', 'Piece"', 'in')
('Piece"', 'in', 'order')
('in', 'order', 'to')
('order', 'to', 'become')
('to', 'become', 'the')
('become', 'the', 'next')
('the', 'next', 'King')
('next', 'King', 'of')
('King', 'of', 'the')
('of', 'the', 'Pirates.')



------Trigram Model using text file------
For sentence 1 , trigrams are: 
('One', 'Piece', '(stylized')
('Piece', '(stylized', 'in')
('(stylized', 'in', 'all')
('in', 'all', 'caps)')
('all', 'caps)', 'is')
('caps)', 'is', 'a')
('is', 'a', 'Japanese')
('a', 'Japanese', 'manga')
('Japanese', 'manga', 'series')
('manga', 'series', 'written')
('series', 'written', 'and')
('written', 'and', 'illustrated')
('and', 'illustrated', 'by')
('illustrated', 'by', 'Eiichiro')
('by', 'Eiichiro', 'Oda')

For sentence 2 , trigrams are: 
('The', 'story', 'follows')
('story', 'follows', 'the')
('follows', 'the', 'adventures')
('the', 'adventures', 'of')
('adventures', 'of', 'Monkey')
('of', 'Monkey', 'D')
('Monkey', 'D', 'Luffy')
('D', 'Luffy', 'and')
('Luffy', 'and', 'his')
('and', 'his', 'crew,')
('his', 'crew,', 'the')
('crew,', 'the', 'Straw')
('the', 'Straw', 'Hat')
('Straw', 'Hat', 'Pirates,')
('Hat', 'Pirates,', 'where')
('Pirates,', 'where', 'he')
('where', 'he', 'explores')
('he', 'explores', 'the')
('explores', 'the', 'Grand')
('the', 'Grand', 'Line')
('Grand', 'Line', 'in')
('Line', 'in', 'search')
('in', 'search', 'of')
('search', 'of', 'the')
('of', 'the', 'mythical')
('the', 'mythical', 'treasure')
('mythical', 'treasure', 'known')
('treasure', 'known', 'as')
('known', 'as', 'the')
('as', 'the', '"One')
('the', '"One', 'Piece"')
('"One', 'Piece"', 'in')
('Piece"', 'in', 'order')
('in', 'order', 'to')
('order', 'to', 'become')
('to', 'become', 'the')
('become', 'the', 'next')
('the', 'next', 'King')
('next', 'King', 'of')
('King', 'of', 'the')
('of', 'the', 'Pirates')

'''
