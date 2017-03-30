# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 01:38:32 2017

@author: anbarasan.selvarasu
"""

from deep_learning import language_model
import pickle
import vocabulary
import nltk


documents_pickle = pickle.load(open('processed_inbox_docs_full.pickle', 'rb'))

sample_subset = []
sample_subset.append(documents_pickle['brawner-s'])
sample_subset.append(documents_pickle['blair-l'])

#vocab_dict = vocabulary.get_vocabulary(sample_subset)
DEFAULT_TRAINING_CONFIG = {'batch_size': 100,                    # the size of a mini-batch
                           'learning_rate': 0.1,                 # the learning rate
                           'momentum': 0.9,                      # the decay parameter for the momentum vector
                           'epochs': 2,                         # the maximum number of epochs to run
                           'init_wt': 0.01,                      # the standard deviation of the initial random weights
                           'context_len': 3,                     # the number of context words used
                           'show_training_CE_after': 100,        # measure training error after this many mini-batches
                           'show_validation_CE_after': 1000,     # measure validation error after this many mini-batches
                           }


model_a = language_model.train(50, 100, config=DEFAULT_TRAINING_CONFIG)