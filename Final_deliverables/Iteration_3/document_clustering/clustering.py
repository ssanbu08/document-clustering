# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 06:17:28 2017

@author: anbarasan.selvarasu
"""
from scipy.spatial.distance import cdist        
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics 
from time import time     
from sklearn.decomposition import PCA  
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.manifold import MDS

class Clustering(object):
    
   
    def __init__(self):
        pass
    
    """
    Plot average distance from observations from the cluster centroid
    to use the Elbow Method to identify number of clusters to choose
    """
    def find_k(self,df,cluster_range):
        #df = feature_matrix
        # cluster_range= 3
        kMeansVar = [KMeans(n_clusters=k).fit(df) for k in range(1, cluster_range)]
        centroids = [X.cluster_centers_ for X in kMeansVar]
        k_euclid = [cdist(df.toarray(), cent) for cent in centroids]
        dist = [np.min(ke, axis=1) for ke in k_euclid]
        wcss = [sum(d**2) for d in dist]
        tss = sum(pdist(df.toarray())**2)/df.shape[0]
        bss = tss - wcss
        plt.plot(wcss)
        #plt.plot(bss)
        plt.show()
        
    ##########################################################################
    ####################### Clustering #######################################
    ##########################################################################
    
    def do_clustering(self,num_clusters,data,mode = 'normal'):
        
        if mode == 'minibatch':
            km = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000)
        elif mode == 'normal':
            km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
        else:
            raise Exception("Wrong mode type entered. Possible values: 'normal','minibatch'")
        
        print("Clustering sparse data with %s" % km)
        t0 = time()
        km.fit(data)
        print("done in %0.3fs" % (time() - t0))
        print()
        clusters = km.labels_
        return km

    
    
#    def k_means(feature_matrix, num_clusters=5):
#        km = KMeans(n_clusters=num_clusters,
#                    max_iter=10000)
#        km.fit(feature_matrix)
#        clusters = km.labels_
#        return km, clusters
    
    ##########################################################################
    ##################### Evaluation #########################################
    ##########################################################################
                    
    def evaluate_clustering(self,estimator, name, data):
        t0 = time()
        estimator.fit(data)
        print('% 9s   %.2fs    %i    %.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean')))
            ## this is the only metric available in sklearn
            ## for unsupervised setting
    #####################################################################
    ################ Get Top terms from clustering ######################                                      
    #####################################################################                                          
    def get_top_terms(self,km,svd,num_clusters,vectorizer,factorized =False):
        print("Top terms per cluster:")
        
        if(factorized):
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
                
        terms = vectorizer.get_feature_names()
        for i in range(num_clusters):
            print'Cluster %d:' % i
            for ind in order_centroids[i, :10]:
                print' %s' % terms[ind]
            print()    
    
    #####################################################################
    ##################### Visualization  ################################
    #####################################################################
     
    def visualize_clusters(self,distance_matrix,clusters,emps_index):
        # convert two components as we're plotting points in a two-dimensional plane
        # "precomputed" because we provide a distance matrix
        # we will also specify `random_state` so the plot is reproducible.
        mds = MDS(n_components=2, random_state=1)
        
        pos = mds.fit_transform(distance_matrix)  # shape (n_components, n_samples)
        
        xs, ys = pos[:, 0], pos[:, 1]
        print()
        print() 
        
        
        #set up colors per clusters using a dict
        cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 
                          5: "#d73027", 6: "#fc8d59", 7: "#fee08b", 8: "#d9ef8b", 9: "#91cf60",
                          10: "#1a9850"}
        
        #set up cluster names using a dict
        cluster_names = {0: 'Cluster 1', 
                         1: 'Cluster 2', 
                         2: 'Cluster 3', 
                         3: 'Cluster 4', 
                         4: 'Cluster 5',
                         5: 'Cluster 6', 
                         6: 'Cluster 7', 
                         7: 'Cluster 8', 
                         8: 'Cluster 9',
                         9: 'Cluster 10',
                        10: 'Cluster 11'} 
        
        
        #create data frame that has the result of the MDS plus the cluster numbers and titles
        import pandas as pd
        df = pd.DataFrame(dict(x=xs, y=ys, label= clusters, title=emps_index)) 
        
        #group by cluster
        groups = df.groupby('label')
        
        
        # set up plot
        fig, ax = plt.subplots(figsize=(17, 9)) # set size
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        
        #iterate through groups to layer the plot
        #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
                    label=cluster_names[name], color=cluster_colors[name], 
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params(\
                axis= 'x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            ax.tick_params(\
                axis= 'y',         # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelleft='off')
            
        ax.legend(numpoints=1)  #show legend with only 1 point
        
        #add label in x,y position with the label as the film title
        for i in range(len(df)):
            ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  
        
            
            
        plt.show() #show the plot
        
        #uncomment the below to save the plot if need be
        fig.savefig('clusters_small_noaxes.png', dpi=200)     


    def visualize_clusters_2(self,data,km,num_clusters):
        reduced_data = PCA(n_components=2).fit_transform(data)
        km.fit(reduced_data)
        
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
        
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Obtain labels for each point in mesh. Use last trained model.
        Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')
        
        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = km.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                  'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


###############################################################################################
################# TESTING ############################
'''

from cls_cluster_validation import *

num_clusters = 2
objClus = Clustering()

### Step 1: Fixing K ########
objClus.find_k(feature_matrix,2)



### Step 2: Clustering ######

normal_kmeans = objClus.do_clustering(num_clusters,feature_matrix)
minibatch_kmeans = objClus.do_clustering(num_clusters,feature_matrix,'minibatch')

print('Saving Kmeans Model as pickle')
import pickle
with open('normal_kmeans.pickle', 'wb') as handle:
    pickle.dump(normal_kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('normal_kmeans.pickle', 'rb') as handle:
    normal_kmeans = pickle.load(handle)

### Step 3: Evaluate Cluster ####
  
objClus.evaluate_clustering(normal_kmeans,'initial iteration',feature_matrix)  

### Step 4: Visualize Clusters ###
objDist = Distance()
dist_cosine  = objDist.calculate_distance(feature_matrix,mode = 'cosine')

objClus.visualize_clusters(dist_cosine,normal_kmeans.labels_,emp_index)



'''
