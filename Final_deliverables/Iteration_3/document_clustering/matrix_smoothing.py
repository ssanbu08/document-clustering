# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 06:59:44 2017

@author: anbarasan.selvarasu
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
import logging
from time import time

class MatrixSmoothing(object):
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
                    
    def find_lsa(self,data,num_components):
        if num_components:
            print("Performing dimensionality reduction using LSA")
            t0 = time()
            # Vectorizer results are normalized, which makes KMeans behave as
            # spherical k-means for better results. Since LSA/SVD results are
            # not normalized, we have to redo the normalization.
            svd = TruncatedSVD(num_components)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
        
            factorized_data = lsa.fit_transform(data)
        
            print("done in %fs" % (time() - t0))
        
            explained_variance = svd.explained_variance_ratio_.sum()
            print("Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))
        
            print()
            return svd,factorized_data
            
    
            
##########################################################################################
######################## TESTING ########################################################
"""
from cls_matrix_smoothing import *
objMS = MatrixSmoothing()

### STEP 1: Transformation ####
svd = objMS.find_lsa(feature_matrix,100)

### STEP 2: Clustering on Transformed Data ####
num_clusters = 2
objClus = Clustering()

normal_kmeans = objClus.do_clustering(num_clusters,svd)


#### STEP 3: Inverse Transformation ###########
inverse_transformation(normal_kmeans)





"""