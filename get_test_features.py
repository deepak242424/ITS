# -*- coding: utf-8 -*-


import cv2
import numpy as np
import cPickle as cp
import os
from config import  Config

class feature_vectors(object):
    def __init__(self, config):
        self.feature_type = config.feature_type

        self.feature_vectors = []
        self.base_path = config.feat_path
        self.train_data_path = config.test_data_path
        self.clusters = config.clusters
        self.f_scalar = open(self.base_path+'StandardScalar_'+self.feature_type+'_'+'.save', 'rb')
        self.scaler = cp.load(self.f_scalar)
        self.f_scalar.close()

        self.kmeans_file = open(self.base_path+'kmeans_'+self.feature_type+'_'+str(self.clusters)+'.save', 'rb')
        self.kmeans = cp.load(self.kmeans_file)
        self.kmeans_file.close()
        
    def get_cls_features_vec(self, cls_path):
        cls_feat_vecs = []
        for gray in os.listdir(cls_path):
            img = cv2.imread(cls_path+gray)
            detector = cv2.FeatureDetector_create(self.feature_type)
            kp = detector.detect(img)
            sift = cv2.DescriptorExtractor_create(self.feature_type)
            des = sift.compute(img, kp)
            cls_feat_vecs.append(self.get_fea_vec(des[1]))
        self.feature_vectors.append(cls_feat_vecs)
            
    def get_fea_vec(self, keypoints): #input : num of keypoints * 64 or 128 dims
        feature_vec = np.zeros(self.clusters)
        if keypoints != None:
            keypoints = self.scaler.transform(keypoints)
            for i, itr in enumerate(keypoints):
                itr = itr[np.newaxis, :]
                cluster = self.kmeans.predict(itr)[0]
                feature_vec[cluster] += 1
        return feature_vec
 
    def get_all_features_vec(self, classes):
        for path in classes:
            self.get_cls_features_vec(self.train_data_path + path + '/' )
            print "Done with class " + path             

def execute_get_test_features(config):
    feat_types = config.feature_type
    cluster_nums = config.clusters
    
    feat_types = feat_types*5
    cluster_nums = cluster_nums*2
    
    for fea, clus in zip(["SIFT"], [128]):
        config.set_feat_cluster(fea, clus)
        feat_vects = feature_vectors(config)
        
        classes = config.classes
        feat_vects.get_all_features_vec(classes)
        
        pickle_file = open(config.feat_path+config.feature_type+"_test_feat_vecs_"+str(config.clusters)+".save", 'wb')
        cp.dump(feat_vects.feature_vectors, pickle_file, protocol=cp.HIGHEST_PROTOCOL)
        pickle_file.close()          
        
        print "Done with test features "#, fea, clus
        
if __name__ == 'main':
    config = Config()
    execute_get_test_features(config)
        
    