# POS-Tagging-with-Viterbi-Decoding

An HMM model for part of speech taggging using viterbi decoding. 

Dependecies:
numpy

Program Details:

The HMM is initialized on a bigram model of the words/tags. The attributes are as follows:

Dictionary: a dictionary of the vocabulary, k=word, v=count

Transitions: a 2d dictionary where transitions[tag_(i-1)][tag_i] = count

Emissions: a 2d dictionary where emissions[tag][word] = count

Context: a dictionary of tags where k=tag, v=count

A: transition matrix; a 2d KxK matrix (such that K is the cardinality of tags); 
each transition probability takes both the number of times the tag-tag transition occurs, 
as well as the context aka the number of times a tag occurs in the corpus

B: emission matrix; a 2d KxN matrix (such that N is the length of Dictionary); 
each emission probability takes both the number of times the tag-word association is observed, 
as well as the context aka the number of times a tag occurs in the corpus; smoothing is used


The Viterbi algorithm has the following variables:

VM: the Viterbi matrix; a 2d KxT matrix (such that K is the cardinality of tags and T is the 
length of the words to be tagged, called data); 
each entry is the maximum a priori estimate (prior * likelihood) of a tag given the word

BpM: the backpointer matrix; a 2d KxT matrix; a matrix that keeps track of the index 
associated with the maximum probability calculated for the vector of tags associated with a word

Keeping track of the MAP probabilities of p(t|w) for each word in the input data, 
and the associated path of those maximum probabilities, we backtrack through BpM 
to construct the optimal path and associated tags.


consulted sources: 
https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
https://web.stanford.edu/~jurafsky/slp3/8.pdf
