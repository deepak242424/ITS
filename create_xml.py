# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:42:53 2017

@author: deepack
"""

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
import xml.etree.ElementTree as ET

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

        self.class_map = {'Ace' : 'Car', 'Activa' : 'Person', 'Auto' : 'Car', 
                          'Bike' : 'Person', 'Bus' : 'Bus', 'Car' : 'Car', 'Jeep' : 'Car', 
                          'Pedstn' : 'Person', 'Sedan' : 'Car', 'Traveller' : 'Bus', 
                          'Truck' : 'Bus', 'Van' : 'Car'}
        self.dic_count = {'Car' : 0, 'Bus' : 0, 'Person' : 0}
#        ---------- 4 class--------------
        if config.num_classes == 4:
            self.class_map = {'Ace' : 'Auto', 'Activa' : 'Person', 'Auto' : 'Auto', 
                              'Bike' : 'Person', 'Bus' : 'Bus', 'Car' : 'Car', 'Jeep' : 'Car', 
                              'Pedstn' : 'Person', 'Sedan' : 'Car', 'Traveller' : 'Bus', 
                              'Truck' : 'Bus', 'Van' : 'Car'}
            self.dic_count = {'Car' : 0, 'Auto' : 0, 'Bus' : 0, 'Person' : 0}
#       -----------5 class------------------     
        elif config.num_classes == 5:
            self.class_map = {'Ace' : 'Auto', 'Activa' : 'Activa', 'Auto' : 'Auto', 
                          'Bike' : 'Person', 'Bus' : 'Bus', 'Car' : 'Car', 'Jeep' : 'Car', 
                          'Pedstn' : 'Person', 'Sedan' : 'Car', 'Traveller' : 'Bus', 
                          'Truck' : 'Bus', 'Van' : 'Car'}                      
            self.dic_count = { 'Activa' : 0, 'Car' : 0, 'Auto' : 0, 'Bus' : 0, 'Person' : 0}                    
        
        self.classes_modif = {}
        self.data = None
        
        self.dims = config.dims
        self.interpolation = config.interpolation
        self.base_path = config.frcnn_data 
#        self.base_path = '/home/deepack/Documents/amNotKidding/ieee/data_4class_added_size/'
#        self.base_path = '/home/deepack/Documents/amNotKidding/ieee/data_test/'
        
        self.path_dic = {}
        self.check_paths()
        self.frame_count = 0
        self.trainVal = open(self.base_path+'trainval.txt', 'w')
        self.test = open(self.base_path+'test.txt', 'w')
        
        self.bus_count=0
        self.non_bus_count =0
        
        self.map_json_xml_test = {} 
        self.map_json_xml_train = {}
        
        for cls in self.classes:
            self.classes_modif[cls] = self.class_map[cls.split('-')[0]]  

    def resize_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(img, self.dims, interpolation = self.interpolation)
        return res
 
    def get_json_data(self, json_file):
        print json_file
        with open(json_file) as jfile:
            self.data = json.load(jfile)
        return self.data
        
    def check_paths(self):
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        if not os.path.exists(self.base_path + 'images'):
            os.mkdir(self.base_path + 'images')
            os.mkdir(self.base_path + 'xmls')
            os.mkdir(self.base_path + 'images_test')
            os.mkdir(self.base_path + 'xmls_test')
    
    def create_data(self, filename):
        #print filename
        data = self.get_json_data(filename)
        for entry in data :
            if os.path.exists(entry['filename']):
                if len(entry['annotations']) > 0:
                    bus_flag=False
                    root = ET.Element("annotation")
                    ET.SubElement(root, "filename").text="frame_"+str(self.frame_count)+".jpg"#entry['filename'].split('/')[-1]                
                    im = cv2.imread(entry['filename'])
                    
                    temp_2 = entry['filename'].split('/')[-1].split('.')[0]
                    temp_key = temp_2[5:] #filename starts with 'scene'
                
                    root2 = ET.SubElement(root, "size")
                    ET.SubElement(root2, "height").text=str(im.shape[0])
                    ET.SubElement(root2, "width").text=str(im.shape[1])
                    ET.SubElement(root2, "depth").text=str(3)
                    TEMP_FLAG = False                    
                    for itr in range(len(entry['annotations'])):
                        temp_dic = entry['annotations'][itr]
                        if temp_dic['width']*temp_dic['height']>1000:
                            object_t = ET.SubElement(root, "object")
                            TEMP_FLAG =True
                            modif_class = self.classes_modif[temp_dic['class']]
                            if modif_class=='Bus':
                                bus_flag=True
                                self.bus_count+=1
                            ET.SubElement(object_t, "name").text=modif_class
                            self.dic_count[modif_class] += 1
                            ET.SubElement(object_t, "difficult").text=str(0)
                            bnd_box  = ET.SubElement(object_t, "bndbox")
                            ET.SubElement(bnd_box, "xmin").text=str(temp_dic['x'])
                            
                            ET.SubElement(bnd_box, "ymin").text=str(temp_dic['y'])
                            ET.SubElement(bnd_box, "xmax").text=str(temp_dic['x']+temp_dic['width'])
                            ET.SubElement(bnd_box, "ymax").text=str(temp_dic['y']+temp_dic['height'])
    
    #                    car_file = self.base_path+ modif_class +"/" + str(self.dic_count[modif_class])+".jpg"
    #                    im_crop = im[temp_dic['y']:temp_dic['y']+temp_dic['height'],\
    #                                             temp_dic['x']:temp_dic['x']+temp_dic['width'],:]
    #                    im_crop = self.resize_img(im_crop)
    #                    self.dic_count[modif_class] = self.dic_count[modif_class] + 1
    #                    cv2.imwrite(car_file, im_crop)
                    if bus_flag==False:
                        self.non_bus_count+=1
                    if TEMP_FLAG:                        
                        if (bus_flag and self.bus_count<230) or (bus_flag==False and self.non_bus_count<1000):
                            tree = ET.ElementTree(root)
                            tree.write(self.base_path+'xmls/' + "frame_"+str(self.frame_count)+".xml")
                            cv2.imwrite(self.base_path+'images/' + "frame_"+str(self.frame_count)+".jpg", im)
                            self.trainVal.write("frame_"+str(self.frame_count) + '\n')
                            self.map_json_xml_train[self.frame_count] = temp_key
                        else:
                            tree = ET.ElementTree(root)
                            tree.write(self.base_path+'xmls_test/' + "frame_"+str(self.frame_count)+".xml")
                            cv2.imwrite(self.base_path+'images_test/' + "frame_"+str(self.frame_count)+".jpg", im)
                            self.map_json_xml_test[self.frame_count] = temp_key
                            self.test.write("frame_"+str(self.frame_count) + '\n')
                        self.frame_count += 1
                    
                    im = None
    
    def create_data_from_files(self, lis_filenames):
        for filename in lis_filenames:
            self.create_data(filename)
        self.trainVal.close()
        self.test.close()

def execute_create_xml(config):
    genData = gen_data(config)   
    print genData.non_bus_count,genData.bus_count        
    #json_files = ["/home/deepack/Documents/civil/dataset/curated_by_avinash/Dataset/new_modif_final.json"]
    
    json_folder = '/home/deepack/Documents/civil/dataset/json_files'
    json_files = os.listdir(json_folder)
    json_files = [os.path.join(json_folder, f) for f in json_files]
    #print json_files
    genData.create_data_from_files(json_files)
    
    with open(config.base_path + 'log.txt', 'a') as f:
        f.writelines('Total Counts from xmls ' + str(genData.dic_count) + '\n')
    
    import cPickle as cp
    f_map = open(config.frcnn_data + 'map_json_xml.save', 'wb')
    cp.dump([genData.map_json_xml_train, genData.map_json_xml_test], f_map, cp.HIGHEST_PROTOCOL)
    f_map.close()

if __name__ == "__main__":
    config = Config() 
    execute_create_xml(config)

    