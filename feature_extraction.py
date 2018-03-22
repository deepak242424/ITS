# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:00:16 2017

@author: deepack
"""

import cv2
import numpy as np
import cPickle as cp
import os
from config import  Config
from standard_scalar import mystandard_scaler

class extract_features(object):
    def __init__(self, config):
        self.feature_type = config.feature_type
        self.all_features = []
        self.feat_path = config.feat_path
        
    def get_cls_features(self, cls_path):
        for gray in os.listdir(cls_path):
            img = cv2.imread(cls_path+gray)
            detector = cv2.FeatureDetector_create(self.feature_type)
            kp = detector.detect(img)
            sift = cv2.DescriptorExtractor_create(self.feature_type)
            des = sift.compute(img, kp)
            self.all_features.append(des[1])
 
    def get_all_features(self, train_data_path, classes):
        for path in classes:
            self.get_cls_features(train_data_path + path + '/' )
            print "Done with class " + path     

def execute_feature_extraction(config):
    extract_feats = extract_features(config)
    
    classes = config.classes
    train_data_path = config.train_data_path
    
    extract_feats.get_all_features(train_data_path, classes)
    print len(extract_feats.all_features)
    
    pickle_file = open(config.feat_path+config.feature_type+".save", 'wb')
    cp.dump(extract_feats.all_features, pickle_file, protocol=cp.HIGHEST_PROTOCOL)
    pickle_file.close()  
    
    mystandard_scaler(config)

if __name__=="__main__":
    
    config = Config()
    