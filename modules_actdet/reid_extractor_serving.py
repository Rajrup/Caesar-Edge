import numpy as np 
from os.path import join 
import os 
import sys 
from time import time 
from modules_actdet.data_reader import DataReader
from modules_actdet.data_writer import DataWriter

# # Yitao-TLS-Begin
# import os
# import sys
# from tensorflow.python.saved_model import builder as saved_model_builder
# from tensorflow.python.saved_model import signature_constants
# from tensorflow.python.saved_model import signature_def_utils
# from tensorflow.python.saved_model import tag_constants
# from tensorflow.python.saved_model import utils
# from tensorflow.python.util import compat

# tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
# FLAGS = tf.app.flags.FLAGS

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util
# # Yitao-TLS-End

# # Download the model file to 'checkpoints/'
# DEEPSORT_MODEL = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/checkpoints/deepsort/mars-small128.pb'

DS_HOME = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/modules_actdet/deep_sort'
sys.path.insert(0, DS_HOME)
# The original DS tools folder doesn't have init file, add it
fout = open(join(DS_HOME, 'tools/__init__.py'), 'w')
fout.close()
from tools.generate_detections_serving import create_box_encoder

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
class FeatureExtractor:
    ds_boxes = []
    input = {}
    features = []

    def Setup(self):
        # self.encoder = create_box_encoder(DEEPSORT_MODEL, batch_size=16)
        # self.log('init')
        pass


    def PreProcess(self, input):
        ichannel = grpc.insecure_channel("localhost:8500")
        self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

        self.encoder = create_box_encoder(self.istub, batch_size=16)

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
            return ouptut

        for i in range(len(self.ds_boxes)):
            output['meta']['obj'][i]['box'] = self.ds_boxes[i]
            output['meta']['obj'][i]['feature'] = self.features[i]
            
        return output 


    def log(self, s):
        print('[FExtractor] %s' % s)

