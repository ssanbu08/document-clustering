# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:26:08 2017

@author: anbarasan.selvarasu
"""


import pandas
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from gensim import corpora, models, similarities, matutils
from utility_func import normalization
from scipy.spatial import distance


class LDATopicModeling(object):

    def __init__(self):
        print("Object Instantiated Successfully")
    
    # This method is used to analyse how many num_topics can be choosen to the model with perplexity (predictive likelihood) measure        
    def analyse_num_topics(self,corpus, id2word_dict, start = 1, end = 15, step_size=1):
            
         #data_path = "D:\Enron\code\Iteration_3"
        
        # split into train and test - random sample, but preserving order
        train_size = int(round(len(corpus)*0.8))
        train_index = sorted(random.sample(range(len(corpus)), train_size))
        test_index = sorted(set(range(len(corpus)))-set(train_index))
        train_corpus = [corpus[i] for i in train_index]
        test_corpus = [corpus[j] for j in test_index]
    
        
        grid = defaultdict(list)
        number_of_words = sum(cnt for document in test_corpus for _, cnt in document)
        parameter_list = range(start, end, step_size)
        for parameter_value in parameter_list:
            print ("starting pass for parameter_value = %.3f" % parameter_value)
           # model = models.LdaMulticore(corpus=bow_corpus, workers=None, id2word=dictionary, num_topics=parameter_value, iterations=10)
            model = models.LdaModel(train_corpus, num_topics=parameter_value, id2word=id2word_dict, update_every=1, passes=5, iterations=500)
            perplex = model.bound(test_corpus) # this is model perplexity not the per word perplexity
            print ("Total Perplexity: %s" % perplex)
            grid[parameter_value].append(perplex)
            
            per_word_perplex = np.exp2(-perplex / number_of_words)
            print ("Per-word Perplexity: %s" % per_word_perplex)
            grid[parameter_value].append(per_word_perplex)
            
           # model.save(data_path + 'ldaMulticore_i10_T' + str(parameter_value) + '_training_corpus.lda')
            print()
        
        for numtopics in parameter_list:
            print (numtopics, '\t',  grid[numtopics])
        
        df = pandas.DataFrame(grid)
        ax = plt.figure(figsize=(7, 4), dpi=300).add_subplot(111)
        df.iloc[1].transpose().plot(ax=ax,  color="#254F09")
        plt.xlim(parameter_list[0], parameter_list[-1])
        plt.ylabel('Perplexity')
        plt.xlabel('topics')
        plt.title('')
        plt.savefig('gensim_multicore_i10_topic_perplexity.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
        plt.show()
        #df.to_pickle(data_path + 'gensim_multicore_i10_topic_perplexity.df')
    
    
    
    
    
    # Method Definition to build LDA topic modeling
    def train_lda_model_gensim(self,corpus, dictionary, total_topics=5):
        
        
     #   tfidf = models.TfidfModel(mapped_corpus)
     #   corpus_tfidf = tfidf[mapped_corpus]
        lda = models.LdaModel(corpus, 
                              id2word=dictionary,
                              iterations=100,
                              num_topics=total_topics)
        return lda                     
    
    
    
    
    # Method Definition to print the topics from LDA model
    def print_topics_gensim(self,topic_model, total_topics=5,
                            weight_threshold=0.0001,
                            display_weights=False,
                            num_terms=None):
        
        for index in range(total_topics):
            topic = topic_model.show_topic(index)
            topic = [(word, round(wt,2)) 
                     for word, wt in topic 
                     if abs(wt) >= weight_threshold]
            if display_weights:
                print('Topic #'+str(index+1)+' with weights')
                print(topic[:num_terms] if num_terms else topic)
            else:
                print('Topic #'+str(index+1)+' without weights')
                tw = [term for term, wt in topic]
                print( tw[:num_terms] if num_terms else tw)
            print()
        
    
    
    # Model Evaluation Method
    def intra_inter(self,model, corpus, freq_below, freq_above, num_pairs=130):
        norm_tokenized_corpus = normalize_corpus(corpus, tokenize=True)
        dictionary = corpora.Dictionary(norm_tokenized_corpus)
        dictionary.filter_extremes(no_below=freq_below, no_above=freq_above)
        
        # split each test document into two halves and compute topics for each half
        part1 = [model[dictionary.doc2bow(tokens[: int(len(tokens) / 2)])] for tokens in norm_tokenized_corpus]
        part2 = [model[dictionary.doc2bow(tokens[int(len(tokens) / 2) :])] for tokens in norm_tokenized_corpus]
        
        # print computed similarities (uses cossim)
        print("average cosine similarity between corresponding parts (higher is better):")
        print(np.mean([matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))
    
        random_pairs = np.random.randint(0, len(norm_tokenized_corpus), size=(num_pairs, 2))
        print("average cosine similarity between 130 random parts (lower is better):")    
        print(np.mean([matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs]))
    
    
    
    # define finding closest document method
    def closest_docs(self,model, corpus,num_topics,doc_id):
        topics = [model[bow_doc] for bow_doc in corpus]
        
        
        # create topic vector (Matrix of topics)
        dense = np.zeros((len(topics), num_topics), float)
        for doc_indx, doc_topics in enumerate(topics):
            for topic_indx, topic_prob in doc_topics:
                dense[doc_indx,topic_indx] = topic_prob
                
        # compute pairwise distances
        pairwise_dist = distance.squareform(distance.pdist(dense))
        # set the diagonal elements of the distance matrix to a high value 
        # it just needs to be larger than the other values in the matrix
        largest = pairwise_dist.max()
        for doc_indx in range(len(topics)):
            pairwise_dist[doc_indx,doc_indx] = largest + 1
        
        return pairwise_dist[doc_id].argmin()
