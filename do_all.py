# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:23:57 2017

@author: deepack
"""

from kmeans_feat_vecs import do_kmeans, get_feat_vecs
from svm_multiclass import execute_svm_multiclass
from config import Config
from create_xml import execute_create_xml
from resize import execute_resize
from feature_extraction import execute_feature_extraction
from get_test_features import execute_get_test_features
from shutil import copytree

if __name__ == "__main__":
    config = Config()
    for path in config.all_paths:
        config.check_path(path)    
    copytree(config.root_dir + 'Background/test/', config.test_data_path+ 'Background/')        
    copytree(config.root_dir + 'Background/train/', config.train_data_path+ 'Background/')        
    
    execute_create_xml(config)
    execute_resize(config)
    config.TRAIN=False
    execute_resize(config)
    execute_feature_extraction(config)
    
    feat_types = ["SIFT"]#, "SURF"]
    cluster_nums = [128]#32, 64, 128, 192, 256]
    
    feat_types = feat_types*len(cluster_nums)
    cluster_nums = cluster_nums*len(feat_types)
    
    for fea, clus in zip(feat_types, cluster_nums):
        config.set_feat_cluster(fea, clus)
        do_kmeans(config)    
        get_feat_vecs(config)
        execute_get_test_features(config)
        execute_svm_multiclass(config)
        print "Done with ", fea, clus
    

#    clf = train_svm(config, dataset)
#    save_clf(config, clf)