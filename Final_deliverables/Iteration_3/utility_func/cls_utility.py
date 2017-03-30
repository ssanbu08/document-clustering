# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 08:24:57 2017

@author: anbarasan.selvarasu
"""

class utils(object):
    
    def __init__(self):
        pass
    
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    def build_feature_matrix(documents, feature_type='frequency',
                             ngram_range=(1, 1), min_df=1, max_df=1.0):
    
        feature_type = feature_type.lower().strip()  
        
        if feature_type == 'binary':
            vectorizer = CountVectorizer(binary=True, min_df=min_df,
                                         max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'frequency':
            vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                         max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'tfidf':
            print "mindf",min_df
            print "max_df", max_df
            vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                         ngram_range=ngram_range)
        else:
            raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")
    
        feature_matrix = vectorizer.fit_transform(documents).astype(float)
        
        return vectorizer, feature_matrix
        
    
        
        