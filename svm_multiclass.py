# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 03:26:03 2017

@author: deepack
"""
from config import Config
import cPickle as cp
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

'''
Best : clusters = 192, feat = 'SIFT'
train_accuracy:  0.973262972735
test accuracy:  0.926746166951

SVC(C=26.5, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
'''

def get_X_and_Y(config, train=True):
    path = config.feat_path+config.feature_type
    if train==True:
        pickle_file = open(path+"_feat_vecs"+'_'+str(config.clusters)+".save", 'rb')
    else:
        pickle_file = open(path+"_test_feat_vecs_"+str(config.clusters)+".save", 'rb')
    feats = cp.load(pickle_file)
    pickle_file.close()    
    #['Background', 'Bike', 'Bus', 'Car']
    
    X = []
    Y = []
    for itr, val in enumerate(feats):
        X.extend(val)
        Y.extend([itr]*len(val))
    dataset = {}
    dataset['data'] = np.asarray(X)
    dataset['target'] = np.asarray(Y)
    return dataset

def train_svm(config, dataset):
#    Cs = np.array([100, 10,1,0.1,0.01,0.001])
#    kernels= ['linear','rbf', 'poly']
#    degree = [2,3,4]
    class_weights = ['balanced']
    #grid_params = dict(C=Cs,kernel= kernels, class_weight = class_weights, degree=degree)
    grid_params = dict(C=[100, 25, 10, 1],kernel= ['rbf'], class_weight = class_weights)
    model = SVC(probability=True)
    grid = GridSearchCV(estimator=model, param_grid=grid_params, n_jobs= -1)
    grid.fit(dataset['data'], dataset['target'])
    #print(grid)
    clf = grid.best_estimator_    
    Y_hat = clf.predict(dataset['data'])
    train_accuracy = np.sum(dataset['target']==Y_hat)/float(len(dataset['data']))
    
    return clf, train_accuracy

def save_clf(config, clf):
    f_open = open(config.models+config.feature_type+'_'+str(config.clusters)+".save", 'wb')
    cp.dump(clf, f_open, cp.HIGHEST_PROTOCOL)
    f_open.close()    
    
def execute_svm_multiclass(config):
    dataset = get_X_and_Y(config, train=True)
    clf, train_accuracy = train_svm(config, dataset)
    print 'train_accuracy: ', train_accuracy
    test_dataset = get_X_and_Y(config, train=False)
    test_Y_hat = clf.predict(test_dataset['data']) 
    test_accuracy = np.sum(test_dataset['target']==test_Y_hat)/float(len(test_dataset['data']))
    print 'test accuracy: ', test_accuracy
    save_clf(config, clf)
    
    with open(config.base_path + 'log.txt', 'a') as f:
        f.writelines('Clusters: ' + str(config.clusters) + ' Feature Type: ' + config.feature_type +'\n')
        f.writelines('Train Accuracy: ' + str(train_accuracy) + '\n')
        f.writelines('Test Accuracy: ' + str(test_accuracy) + '\n')
        
if __name__ == "__main__":
    config = Config()
    execute_svm_multiclass(config)
    

