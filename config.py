import cv2
import os

class Config:
    def __init__(self):
        self.dims = (224, 224)
        self.interpolation = cv2.INTER_AREA

        self.classes = [ 'Auto', 'Background', 'Bike', 'Bus', 'Car']
        self.root_dir = '/home/deepack/Documents/amNotKidding/its/' 
        self.base_path = '/home/deepack/Documents/amNotKidding/its/classes_'\
                                                            +str(len(self.classes))+'/'
        self.feature_type = 'SIFT'
        self.clusters = 128
        
        self.nondeep = self.base_path + 'dataset/nondeep/'
        self.frcnn_data = self.base_path + 'dataset/frcnn/'
        self.train_data_path = self.nondeep + 'train/'
        self.test_data_path = self.nondeep + 'test/'
        self.feat_path = self.base_path + 'features/'
        self.models = self.feat_path + 'trained_models/'
        
        self.all_paths = [self.nondeep, self.train_data_path, self.test_data_path, 
                          self.frcnn_data, self.feat_path, self.models]
        
        self.num_classes = len(self.classes) - 1 # -1 for background
        self.TRAIN = True
        
    def set_feat_cluster(self, feat_type, clusters):
        self.feature_type = feat_type
        self.clusters = clusters
        
    def check_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print 'Generate Path : ', path
            
if __name__=='__main__':
    config = Config()
    for path in config.all_paths:
        config.check_path(path)
    
    