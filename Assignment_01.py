'''
Name : Rahul Kailas jagdhane 
Roll No: 28
Assignment No :01
Title: Text Processing using Spacy Library (Tokenization, Stop Words, Lemmitization, POS tagging)        
'''

# importing libs
import spacy

# loading model for english language
nlp = spacy.load("en_core_web_sm")

# input
inp=(
 "Our adventure is not done yet!"
    "Our adventure is not done yet!"
    "No matter how hard you kick me, it is you who is going to be in pain, not me."
    "Human strength lies in the ability to change yourself."
    "That Is My Ninja Way!"
)
print("Natural Language Processing Practical using Spacy Library\n\n\n")

print("Input: ",inp,"\n\nApplying NLP Operations")
# passing inp to english model
passedtxt=nlp(inp)

# Operation 1 : Tokenization 
print("\n\nOperation 1: Tokenization\n")
for token in passedtxt:
    print(token,token.idx)



#Operation 2: Stop Words
print("\n\nOperation 2: Stop Words\n")

# loading stopwords using spacy library
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
len(spacy_stopwords)

# for stop_word in list(spacy_stopwords)[:10]:
#     print(stop_word)
print("Without not keyword (Other than Stop Words):\n",[token for token in passedtxt if token.is_stop])   #test without stopwords
print("\n\nWith not keyword (Actual Stop Words):\n",[token for token in passedtxt if not token.is_stop]) #actual stop words



#Operation 3: Lemmitization
print("\n\nOperation 3: Lemmitization\n")
for token in passedtxt:
    # using lemma_ to identify the lemma of particular word that have actual dicitionary meaning
    if str(token) != str(token.lemma_):
        print(f"{str(token):>20} : {str(token.lemma_)}")


#Operation 3: Part of Speech Tagging
print("\n\nOperation 4: Part of Speech Tagging\n")
# using tag_ from spacy to tag Part of Speech and pos_ for identifying type of Part of Speech
for token in passedtxt:
    print(
        f"""
TOKEN: {str(token)}
=====
TAG: {str(token.tag_):10} POS: {token.pos_}   
EXPLANATION: {spacy.explain(token.tag_)}"""    
    )        



'''
OUTPUT::

Natural Language Processing Practical using Spacy Library



Input:  Our adventure is not done yet!Our adventure is not done yet!No matter how hard you kick me, it is you who is going to be in pain, not me.Human strength lies in the ability to change yourself.That Is My Ninja Way! 

Applying NLP Operations


Operation 1: Tokenization

Our 0
adventure 4
is 14
not 17
done 21
yet!Our 26
adventure 34
is 44
not 47
done 51
yet!No 56
matter 63
how 70
hard 74
you 79
kick 83
me 88
, 90
it 92
is 95
you 98
who 102
is 106
going 109
to 115
be 118
in 121
pain 124
, 128
not 130
me 134
. 136
Human 137
strength 143
lies 152
in 157
the 160
ability 164
to 172
change 175
yourself 182
. 190
That 191
Is 196
My 199
Ninja 202
Way 208
! 211


Operation 2: Stop Words

Without not keyword (Other than Stop Words):
 [Our, is, not, done, is, not, done, how, you, me, it, is, you, who, is, to, be, in, not, me, in, the, to, yourself, That, Is, My]


With not keyword (Actual Stop Words):
 [adventure, yet!Our, adventure, yet!No, matter, hard, kick, ,, going, pain, ,, ., Human, strength, lies, ability, change, ., Ninja, Way, !]


Operation 3: Lemmitization

                 Our : our
                  is : be
                done : do
                  is : be
                done : do
                  me : I
                  is : be
                  is : be
               going : go
                  me : I
               Human : human
                lies : lie
                That : that
                  Is : be
                  My : my
                 Way : way


Operation 4: Part of Speech Tagging


TOKEN: Our
=====
TAG: PRP$       POS: PRON   
EXPLANATION: pronoun, possessive

TOKEN: adventure
=====
TAG: NN         POS: NOUN   
EXPLANATION: noun, singular or mass

TOKEN: is
=====
TAG: VBZ        POS: AUX   
EXPLANATION: verb, 3rd person singular present

TOKEN: not
=====
TAG: RB         POS: PART   
EXPLANATION: adverb

TOKEN: done
=====
TAG: VBN        POS: VERB   
EXPLANATION: verb, past participle

TOKEN: yet!Our
=====
TAG: NNP        POS: PROPN   
EXPLANATION: noun, proper singular

TOKEN: adventure
=====
TAG: NN         POS: NOUN   
EXPLANATION: noun, singular or mass

TOKEN: is
=====
TAG: VBZ        POS: AUX   
EXPLANATION: verb, 3rd person singular present

TOKEN: not
=====
TAG: RB         POS: PART   
EXPLANATION: adverb

TOKEN: done
=====
TAG: VBN        POS: VERB   
EXPLANATION: verb, past participle

TOKEN: yet!No
=====
TAG: NNP        POS: PROPN   
EXPLANATION: noun, proper singular

TOKEN: matter
=====
TAG: NN         POS: NOUN   
EXPLANATION: noun, singular or mass

TOKEN: how
=====
TAG: WRB        POS: SCONJ   
EXPLANATION: wh-adverb

TOKEN: hard
=====
TAG: RB         POS: ADV   
EXPLANATION: adverb

TOKEN: you
=====
TAG: PRP        POS: PRON   
EXPLANATION: pronoun, personal

TOKEN: kick
=====
TAG: VBP        POS: VERB   
EXPLANATION: verb, non-3rd person singular present

TOKEN: me
=====
TAG: PRP        POS: PRON   
EXPLANATION: pronoun, personal

TOKEN: ,
=====
TAG: ,          POS: PUNCT   
EXPLANATION: punctuation mark, comma

TOKEN: it
=====
TAG: PRP        POS: PRON   
EXPLANATION: pronoun, personal

TOKEN: is
=====
TAG: VBZ        POS: AUX   
EXPLANATION: verb, 3rd person singular present

TOKEN: you
=====
TAG: PRP        POS: PRON   
EXPLANATION: pronoun, personal

TOKEN: who
=====
TAG: WP         POS: PRON   
EXPLANATION: wh-pronoun, personal

TOKEN: is
=====
TAG: VBZ        POS: AUX   
EXPLANATION: verb, 3rd person singular present

TOKEN: going
=====
TAG: VBG        POS: VERB   
EXPLANATION: verb, gerund or present participle

TOKEN: to
=====
TAG: TO         POS: PART   
EXPLANATION: infinitival "to"

TOKEN: be
=====
TAG: VB         POS: AUX   
EXPLANATION: verb, base form

TOKEN: in
=====
TAG: IN         POS: ADP   
EXPLANATION: conjunction, subordinating or preposition

TOKEN: pain
=====
TAG: NN         POS: NOUN   
EXPLANATION: noun, singular or mass

TOKEN: ,
=====
TAG: ,          POS: PUNCT   
EXPLANATION: punctuation mark, comma

TOKEN: not
=====
TAG: RB         POS: PART   
EXPLANATION: adverb

TOKEN: me
=====
TAG: PRP        POS: PRON   
EXPLANATION: pronoun, personal

TOKEN: .
=====
TAG: .          POS: PUNCT   
EXPLANATION: punctuation mark, sentence closer

TOKEN: Human
=====
TAG: JJ         POS: ADJ   
EXPLANATION: adjective (English), other noun-modifier (Chinese)

TOKEN: strength
=====
TAG: NN         POS: NOUN   
EXPLANATION: noun, singular or mass

TOKEN: lies
=====
TAG: VBZ        POS: VERB   
EXPLANATION: verb, 3rd person singular present

TOKEN: in
=====
TAG: IN         POS: ADP   
EXPLANATION: conjunction, subordinating or preposition

TOKEN: the
=====
TAG: DT         POS: DET   
EXPLANATION: determiner

TOKEN: ability
=====
TAG: NN         POS: NOUN   
EXPLANATION: noun, singular or mass

TOKEN: to
=====
TAG: TO         POS: PART   
EXPLANATION: infinitival "to"

TOKEN: change
=====
TAG: VB         POS: VERB   
EXPLANATION: verb, base form

TOKEN: yourself
=====
TAG: PRP        POS: PRON   
EXPLANATION: pronoun, personal

TOKEN: .
=====
TAG: .          POS: PUNCT   
EXPLANATION: punctuation mark, sentence closer

TOKEN: That
=====
TAG: DT         POS: PRON   
EXPLANATION: determiner

TOKEN: Is
=====
TAG: VBZ        POS: AUX   
EXPLANATION: verb, 3rd person singular present

TOKEN: My
=====
TAG: PRP$       POS: PRON   
EXPLANATION: pronoun, possessive

TOKEN: Ninja
=====
TAG: NNP        POS: PROPN   
EXPLANATION: noun, proper singular

TOKEN: Way
=====
TAG: NN         POS: NOUN   
EXPLANATION: noun, singular or mass

TOKEN: !
=====
TAG: .          POS: PUNCT   
EXPLANATION: punctuation mark, sentence closer

'''
