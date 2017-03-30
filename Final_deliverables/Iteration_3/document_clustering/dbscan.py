# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 20:20:08 2017

@author: anbarasan.selvarasu
"""

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
db=DBSCAN(eps=13.0,min_samples=100).fit(X)