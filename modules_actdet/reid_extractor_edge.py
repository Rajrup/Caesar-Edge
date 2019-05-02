import numpy as np 
from os.path import join 
import os 
import sys
import pickle
import cv2
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
import tensorflow as tf
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
    # ds_boxes = []
    # input = {}
    # features = []

    @staticmethod
    def Setup():
        # self.encoder = create_box_encoder(DEEPSORT_MODEL, batch_size=16)
        # self.log('init')
        pass


    def PreProcess(self, request, istub):
        self.istub = istub
        self.encoder = create_box_encoder(self.istub, batch_size=16)

        self.request_input = str(tensor_util.MakeNdarray(request.inputs["client_input"]))
        self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
        self.objdet_output = str(tensor_util.MakeNdarray(request.inputs["objdet_output"]))

        self.ds_boxes = []
        self.scores = []
        for b in self.objdet_output.split('-'):
            tmp = b.split('|')
            b0 = int(tmp[0])
            b1 = int(tmp[1])
            b2 = int(tmp[2])
            b3 = int(tmp[3])
            b4 = float(tmp[4])
            self.ds_boxes.append([b0, b1, b2 - b0, b3 - b1])
            self.scores.append(b4)

    def Apply(self):
        # ''' Extract features and update the tracker 
        # ''' 
        # if not self.input:
        #     return 
        # self.features = self.encoder(self.input['img'], self.ds_boxes)

        self.features = self.encoder(self.image, self.ds_boxes)


    def PostProcess(self):
        features_output = pickle.dumps(self.features)

        next_request = predict_pb2.PredictRequest()
        next_request.inputs['client_input'].CopyFrom(
          tf.make_tensor_proto(self.request_input))
        next_request.inputs['objdet_output'].CopyFrom(
          tf.make_tensor_proto(self.objdet_output))
        next_request.inputs['reid_output'].CopyFrom(
          tf.make_tensor_proto(features_output))

        return next_request

    # def log(self, s):
    #     print('[FExtractor] %s' % s)

