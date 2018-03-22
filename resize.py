# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 23:31:29 2017

@author: deepack
"""

#resize eaach image to 224*224

import cv2
import matplotlib.pyplot as plt
import json
import os
from config import Config
import cPickle as cp

class gen_data(object):
    def __init__(self, config):

        self.classes = ['Ace', 'Activa', 'Auto', 'Bike', 'Bus', 'Car', 
                   'Jeep', 'Pedstn', 'Sedan', 'Traveller', 'Truck', 'Van',
                   'Ace-o1', 'Activa-o1', 'Auto-o1', 'Bike-o1', 'Bus-o1', 'Car-o1', 
                   'Jeep-o1', 'Pedstn-o1', 'Sedan-o1', 'Traveller-o1', 'Truck-o1', 
                   'Van-o1', 'Ace-o2', 'Activa-o2', 'Auto-o2', 'Bike-o2', 'Bus-o2',
                   'Car-o2', 'Jeep-o2', 'Pedstn-o2', 'Sedan-o2', 'Traveller-o2', 
                   'Truck-o2', 'Van-o2', 'Ace-o3', 'Activa-o3', 'Auto-o3', 'Bike-o3',
                   'Bus-o3', 'Car-o3', 'Jeep-o3', 'Pedstn-o3', 'Sedan-o3', 'Traveller-o3',
                   'Truck-o3', 'Van-o3', 'Ace-t1', 'Activa-t1', 'Auto-t1', 'Bike-t1', 
                   'Bus-t1', 'Car-t1', 'Jeep-t1', 'Pedstn-t1', 'Sedan-t1', 'Traveller-t1',
                   'Truck-t1', 'Van-t1']

        self.class_map = {'Ace' : 'Car', 'Activa' : 'Bike', 'Auto' : 'Car', 
                          'Bike' : 'Bike', 'Bus' : 'Bus', 'Car' : 'Car', 'Jeep' : 'Car', 
                          'Pedstn' : 'Bike', 'Sedan' : 'Car', 'Traveller' : 'Bus', 
                          'Truck' : 'Bus', 'Van' : 'Car'}
        self.dic_count = {'Car' : 0, 'Bus' : 0, 'Bike' : 0}                     
#---------------4 class--------   
        if config.num_classes == 4:                          
            self.class_map = {'Ace' : 'Auto', 'Activa' : 'Bike', 'Auto' : 'Auto', 
                              'Bike' : 'Bike', 'Bus' : 'Bus', 'Car' : 'Car', 'Jeep' : 'Car', 
                              'Pedstn' : 'Bike', 'Sedan' : 'Car', 'Traveller' : 'Bus', 
                              'Truck' : 'Bus', 'Van' : 'Car'}
            self.dic_count = {'Car' : 0, 'Auto' : 0, 'Bus' : 0, 'Bike' : 0}
#---------------5 class--------   
        if config.num_classes == 5:
            self.class_map = {'Ace' : 'Auto', 'Activa' : 'Activa', 'Auto' : 'Auto', 
                              'Bike' : 'Bike', 'Bus' : 'Bus', 'Car' : 'Car', 'Jeep' : 'Car', 
                              'Pedstn' : 'Bike', 'Sedan' : 'Car', 'Traveller' : 'Bus', 
                              'Truck' : 'Bus', 'Van' : 'Car'}
            self.dic_count = {'Car' : 0, 'Auto' : 0, 'Bus' : 0, 'Bike' : 0, 'Activa' : 0}                     
    #        self.class_map = {'Ace' : 'Ace', 'Activa' : 'Activa', 'Auto' : 'Auto', 
    #                          'Bike' : 'Bike', 'Bus' : 'Bus', 'Car' : 'Car', 'Jeep' : 'Jeep', 
    #                          'Pedstn' : 'Bike', 'Sedan' : 'Sedan', 'Traveller' : 'Bus', 
    #                          'Truck' : 'Bus', 'Van' : 'Van'}
                          
        self.classes_modif = {}
        self.data = None
#        self.dic_count = {'Car' : 0, 'Auto' : 0, 'Bus' : 0, 'Bike' : 0}                     
        self.dims = config.dims
        self.interpolation = config.interpolation
        self.base_path = config.nondeep 
        self.path_dic = {}
        self.check_paths(config)
        self.config = config
        
        for cls in self.classes:
            self.classes_modif[cls] = self.class_map[cls.split('-')[0]]  

    def resize_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #res = cv2.resize(img, self.dims, interpolation = self.interpolation)
        #return res
        return img
 
    def get_json_data(self, json_file):
        with open(json_file) as jfile:
            self.data = json.load(jfile)
        return self.data
        
    def check_paths(self, config):
        if os.path.exists(self.base_path):
            #os.mkdir(self.base_path)
            if config.TRAIN:
                self.base_path = self.base_path + 'train/'
            else:
                self.base_path = self.base_path+'test/'
            if not os.path.exists(self.base_path+'Bus'):
                for cls in set(self.class_map.values()):
                    os.makedirs(self.base_path+ cls)
                    self.path_dic[cls] = self.base_path+ cls
                        
    def create_data(self, filename, map_json_xml):
        temp_1 = filename.split('/')[-1].split('.')[0]
        data = self.get_json_data(filename)
        for entry in data :
            if os.path.exists(entry['filename']):
#                print entry['filename']
                temp_2 = entry['filename'].split('/')[-1].split('.')[0]
                temp_key = temp_2[5:]
                
                if self.config.TRAIN:
                    json_map = map_json_xml[0].values()
                else:
                    json_map = map_json_xml[1].values()
                if temp_key in json_map:
                    #print temp_key 
                    im = cv2.imread(entry['filename'])    
                    for itr in range(len(entry['annotations'])):
                        #print entry['annotations'][itr]
                        temp_dic = entry['annotations'][itr]
                        if temp_dic['height'] * temp_dic['width'] > 1000 :
                            modif_class = self.classes_modif[temp_dic['class']]
#                            car_file = self.base_path+ modif_class +"/" + str(self.dic_count[modif_class])+"_"+temp_1+"_"+temp_2+".jpg"
                            car_file = self.base_path+ modif_class +"/" + str(self.dic_count[modif_class])+".jpg"
                            im_crop = im[temp_dic['y']:temp_dic['y']+temp_dic['height'],\
                                                     temp_dic['x']:temp_dic['x']+temp_dic['width'],:]
                            im_crop = self.resize_img(im_crop)
                            self.dic_count[modif_class] = self.dic_count[modif_class] + 1
                            cv2.imwrite(car_file, im_crop)
                    im = None
    
    def create_data_from_files(self, lis_filenames, map_json_xml):
        for filename in lis_filenames:
            self.create_data(filename, map_json_xml)

def execute_resize(config):
    genData = gen_data(config)           
    
    f_open = open(config.frcnn_data + 'map_json_xml.save', 'rb')
    map_json_xml = cp.load(f_open)
    f_open.close()
    
    json_folder = '/home/deepack/Documents/civil/dataset/json_files'
    json_files = os.listdir(json_folder)
    json_files = [os.path.join(json_folder, f) for f in json_files]
    print json_files
    genData.create_data_from_files(json_files, map_json_xml)
    
    with open(config.base_path + 'log.txt', 'a') as f:
        if config.TRAIN:
            f.writelines('Train Counts: '+ str(genData.dic_count) + '\n')
        else:
            f.writelines('Test Counts: '+ str(genData.dic_count) + '\n')
    
if __name__ == "__main__":
    config = Config() 
    execute_resize(config)

    