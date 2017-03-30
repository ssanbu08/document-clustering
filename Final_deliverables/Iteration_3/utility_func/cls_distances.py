# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 07:53:49 2017

@author: anbarasan.selvarasu
"""
from sklearn.metrics import pairwise_distances

class Distance(object):
    
    def __init__(self):
        pass
    
    
    def calculate_distance(self,feature_matrix,mode = 'euclidean'):
        if mode == 'cosine':
            return pairwise_distances(feature_matrix,metric='cosine', n_jobs=-2) # n_jobs = -2 uses all but one CPU
        elif mode == 'euclidean':
            return pairwise_distances(feature_matrixmetric='euclidean', n_jobs=-2)
        else:
            raise Exception("Wrong mode type entered. Possible values: 'euclidean', 'cosine'")

        
        
