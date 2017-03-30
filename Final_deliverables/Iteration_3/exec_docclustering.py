# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 10:40:58 2017

@author: anbarasan.selvarasu
"""

#==============================================================================
# % load_ext autoreload
# % autoreload 2
#==============================================================================





from data_preparation import data_preparation 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# reload(data_preparation)

##### STEP 1:  Data Preparation ####

print('Load dataset...')
dp = DataPreparation("C:/Users/anbarasan.selvarasu/Documents/Enron/enron_mail_20150507/maildir")
emails_dict = dp.get_inbox_of()

print emails_dict.keys()
print emails_dict['brawner-s'][0][1] ## One of the emails by brawn-s

##### STEP 2:  Get a Single Document of emails for each employee #####

documents = dp.construct_documents(emails_dict)

##### STEP 3: Save the Loaded document as a pickle file ##########

print('Saving prepared copus as pickle')
import pickle
#pickle.dump(documents, open('processed_inbox_docs_full.pickle', 'wb'))

document_pickle = pickle.load(open('processed_inbox_docs_full.pickle', 'rb'))


#### STEP 4 ##########:
from feature_generation import tf_idf_features
import numpy as np
# reload(tf_idf_features)
### STEP 4.1 #########
'
objFeatGen = tf_idf_features.FeatureGeneration()
# obtained through various trail and error approaches and fixed min_df = 5, max_df = 0.1
vectorizer,feature_matrix =objFeatGen.compute_feature_matrix(document_pickle,min_df = 5,max_df = 0.1)
print feature_matrix.shape

pickle.dump(feature_matrix, open('feat_matrix_5_01.pickle', 'wb'))
vectorizer_word = pickle.dump(vectorizer, open('vectorizer_5_01.pickle', 'wb'))

feature_matrix = pickle.load(open('feat_matrix_5_01.pickle', 'rb'))
vectorizer = pickle.load(open('vectorizer_5_01.pickle', 'rb'))

########## STEP 5 Dimensionality Reduction ###############
from document_clustering import matrix_smoothing 
objMS = matrix_smoothing.MatrixSmoothing()

### STEP 5.1: Transformation ####
svd,lsa = objMS.find_lsa(feature_matrix,137)
print lsa.shape
pickle.dump(svd, open('svd_137.pickle', 'wb'))
pickle.dump(lsa, open('lsa_137.pickle', 'wb'))
svd = pickle.load(open('svd_137.pickle', 'rb'))
lsa = pickle.load(open('lsa_137.pickle', 'rb'))

########## STEP 5.2: Clustering ###########################

from document_clustering import clustering
from scipy import sparse
from utility_func import cls_distances

cluster_range = 10
objClus =clustering. Clustering()

### Step 6.1: Fixing K ########

objClus.find_k(feature_matrix,cluster_range)# feature_matrix should be given here


### Step 6.2: Clustering ######
num_clusters = 6
normal_kmeans_features = objClus.do_clustering(num_clusters,feature_matrix)
normal_kmeans_svd = objClus.do_clustering(num_clusters,lsa)
minibatch_kmeans = objClus.do_clustering(num_clusters,feature_matrix,'minibatch')

print('Saving Kmeans Model as pickle')
import pickle
with open('normal_kmeans.pickle', 'wb') as handle:
    pickle.dump(normal_kmeans, handle)

with open('normal_kmeans.pickle', 'rb') as handle:
    normal_kmeans = pickle.load(handle)
    
### Step 6.3: Evaluate Clustering #####

objClus.evaluate_clustering(normal_kmeans_features,'initial iteration',feature_matrix) 
objClus.evaluate_clustering(normal_kmeans_svd,'initial iteration',lsa)  

### Step 6.4: Visualize Clusters ###
objDist = cls_distances.Distance()
dist_cosine  = objDist.calculate_distance(feature_matrix,mode = 'cosine')

emp_index = objFeatGen.get_emp_index(document_pickle)
objClus.visualize_clusters(lsa,normal_kmeans_features.labels_,emp_index)

objClus.visualize_clusters_2(feature_matrix.toarray(),normal_kmeans_features,num_clusters)

#### Step 6.5: Top terms per cluster #####
objClus.get_top_terms(normal_kmeans_features,svd,num_clusters,vectorizer,factorized =False)
objClus.get_top_terms(normal_kmeans_svd,svd,num_clusters,vectorizer,factorized =True)






