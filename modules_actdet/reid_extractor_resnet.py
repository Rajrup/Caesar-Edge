import numpy as np 
from os.path import join 
import os 
import sys 
from time import time 
from modules_actdet.data_reader import DataReader
from modules_actdet.data_writer import DataWriter

# # Download the model file to 'checkpoints/'
# DEEPSORT_MODEL = './checkpoints/deepsort/mars-small128.pb'

# Config File for the Resnet Model
CONFIG_FILE = "./cfg/config.json"

DS_HOME = './modules_actdet/deep_sort'
sys.path.insert(0, DS_HOME)

# The original DS tools folder doesn't have init file, add it
fout = open(join(DS_HOME, 'tools/__init__.py'), 'w')
fout.close()

from tools.generate_detections_resnet import create_box_encoder2

'''
Input: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'label': label,
                        'conf': conf_score
                        }]
                }
        }

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,w,h],
                        'conf': conf_score
                        'feature': feature_array
                        }]
                }
        }
'''
class FeatureExtractor2:
    ds_boxes = []
    input = {}
    features = []

    def Setup(self):
        self.encoder = create_box_encoder2(CONFIG_FILE, batch_size=16)
        self.log('init')

    def PreProcess(self, input):
        self.input = input
        if not self.input:
            return 

        boxes = input['meta']['obj']
        self.ds_boxes = [[b['box'][0], b['box'][1], b['box'][2] - b['box'][0], 
                                    b['box'][3] - b['box'][1]] for b in boxes]
        
    def Apply(self):
        ''' Extract features and update the tracker 
        ''' 
        if not self.input:
            return 
        self.features = self.encoder(self.input['img'], self.ds_boxes)

    def PostProcess(self):
        output = self.input
        if not self.input:
            return output

        for i in range(len(self.ds_boxes)):
            output['meta']['obj'][i]['box'] = self.ds_boxes[i]
            output['meta']['obj'][i]['feature'] = self.features[i]
            
        return output 

    def log(self, s):
        print('[FExtractor2] %s' % s)