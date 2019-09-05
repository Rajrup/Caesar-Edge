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
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util
# # Yitao-TLS-End

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

    @staticmethod
    def Setup():
        pass

    def PreProcess(self, request_input, istub, grpc_flag):
        self.istub = istub
        self.encoder = create_box_encoder(self.istub, batch_size=16)

        if (grpc_flag):
            self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
            self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
            self.objdet_output = str(tensor_util.MakeNdarray(request_input.inputs["objdet_output"]))
        else:
            self.image = request_input['client_input']
            self.objdet_output = request_input['objdet_output']

        self.ds_boxes = []
        self.scores = []
        # print(self.objdet_output)
        if (self.objdet_output == ""):
          self.objdet_output = "284|110|609|719|0.482117|person"
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


    def PostProcess(self, grpc_flag):
        if (grpc_flag):
            features_output = pickle.dumps(self.features) # @@@ There might be huge overhead here...

            try:
                self.request_input
            except AttributeError:
                self.request_input = cv2.imencode('.jpg', self.image)[1].tostring()
                
            next_request = predict_pb2.PredictRequest()
            next_request.inputs['client_input'].CopyFrom(
              tf.make_tensor_proto(self.request_input))
            next_request.inputs['objdet_output'].CopyFrom(
              tf.make_tensor_proto(self.objdet_output))
            next_request.inputs['reid_output'].CopyFrom(
              tf.make_tensor_proto(features_output))
            return next_request
        else:
            result = dict()
            result['client_input'] = self.image
            result['objdet_output'] = self.objdet_output
            result['reid_output'] = self.features
            # print("[Reid] size of result = %s" % str(sys.getsizeof(result)))
            return result