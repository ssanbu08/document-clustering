# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 10:31:27 2017

@author: anbarasan.selvarasu
"""
import nltk

'''
 This function gives the top n distinct words in the corpus based on word frequencies
 1.This step will be helpful in removing additional stopwords.
 2.this can also be used to pass vocabulary argument to tf_idf vectorizer to get a feature matrix
 of small dimension based upon our choice
'''
def get_vocabulary(corpus):
    word_list=[]
    word_list_flattened = []
    word_freq= []
    for employee,mail in list(corpus.items()):
        tokens = nltk.word_tokenize(mail) 
        tokens = [token.strip() for token in tokens]
        word_list.append(tokens)
    word_list_flattened = [y for x in word_list for y in x]
    for w in word_list_flattened:
        word_freq.append(word_list_flattened.count(w))
    vocab = zip(word_list_flattened, word_freq)
    vocab_dict =  dict(vocab)
    vocab_dict_sort = sorted(vocab_dict, key=vocab_dict.get, reverse=True)

    return vocab_dict_sort
        