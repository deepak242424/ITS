# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 00:00:34 2017

@author: deepack
"""

# -*- coding: utf-8 -*-

import cPickle as cp
from sklearn.cluster import KMeans
from config import Config
from feature_vectors import feature_vectors

def do_kmeans(config):
    f_normed = open(config.feat_path+'norm_'+config.feature_type+'_'+'.save', 'rb')
    X_keypoints = cp.load(f_normed)
    f_normed.close()
    print 'doing kmeans'
    kmeans = KMeans(config.clusters, random_state=0, n_jobs=-2).fit(X_keypoints)

    f_kmeans = open(config.feat_path+'kmeans_'+config.feature_type+'_'+str(config.clusters)+'.save', 'wb')
    cp.dump(kmeans, f_kmeans, protocol=cp.HIGHEST_PROTOCOL)
    f_kmeans.close()

def get_feat_vecs(config):
    feat_vects = feature_vectors(config)
    classes = config.classes
    print "getting feature vectors"
    feat_vects.get_all_features_vec(classes)
    pickle_file = open(config.feat_path+config.feature_type+"_feat_vecs"+'_'+str(config.clusters)+".save", 'wb')
    cp.dump(feat_vects.feature_vectors, pickle_file, protocol=cp.HIGHEST_PROTOCOL)
    pickle_file.close()          

if __name__ == "__main__":
    config = Config()
    do_kmeans(config)    
    get_feat_vecs(config)