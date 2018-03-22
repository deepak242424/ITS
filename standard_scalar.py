# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 02:40:11 2017

@author: deepack
"""

import numpy as np
import cPickle as cp
from sklearn.preprocessing import StandardScaler

from config import Config

def mystandard_scaler(config):
    feature_type = config.feature_type
    pickle_file = open(config.feat_path+feature_type+'.save', 'rb')
    X = cp.load(pickle_file)
    pickle_file.close()
    X_keypoints = []
    
    for val in X:
        if val == None:
            pass
        else:
            for row in val:
                X_keypoints.append(row)
    
    X_keypoints = np.asarray(X_keypoints)
    
    scaler = StandardScaler().fit(X_keypoints)
    f_scalar = open(config.feat_path+'StandardScalar_'+feature_type+'_'+'.save', 'wb')
    cp.dump(scaler, f_scalar, protocol=cp.HIGHEST_PROTOCOL)
    f_scalar.close()
    
    X_keypoints = scaler.transform(X_keypoints)
    
    f_normed = open(config.feat_path+'norm_'+feature_type+'_'+'.save', 'wb')
    cp.dump(X_keypoints, f_normed, cp.HIGHEST_PROTOCOL)
    f_normed.close()

if __name__ == "__main__":
    config = Config()
    mystandard_scaler(config)