# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:26:08 2017

@author: anbarasan.selvarasu
"""

import numpy as np
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class FeatureGeneration(object):
    
    
    def __init__(self):
        pass
    
               
    def compute_feature_matrix(self,document,min_df=1,max_df = 1.0):       
                          
        print 'Document similarity have been reloaded'
        tfidf_vectorizer, tfidf_features = self.build_feature_matrix(document.values(),
                                                            feature_type='tfidf',
                                                            ngram_range=(1, 1), 
                                                            min_df=min_df, max_df=max_df)
                                                            
    #    top_similar_docs = self.compute_cosine_similarity(tfidf_vectorizer,tfidf_features)
        return tfidf_vectorizer,tfidf_features
    
    def build_feature_matrix(self,documents, feature_type='frequency',
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
            
    
    def get_emp_index(self,new_dict):
        emps_index = []
        for i, emp_emails in enumerate(new_dict.items()):
            emps_index.append(emp_emails[0])
            print(emps_index)
        return emps_index
        
    def get_top_similar_docs(self,n=5):
        # get docs with highest similarity scores
        similarity = self.compute_document_similarity()
        top_docs = similarity.argsort()[::-1][:top_n]
        top_docs_with_score = [(index, round(similarity[index], 3))
                                for index in top_docs]
        return top_docs_with_score
            

#==============================================================================
