# Darkflow should be installed from: https://github.com/thtrieu/darkflow
from darkflow.net.build import TFNet
import numpy as np
import tensorflow as tf
from time import time 
from modules_actdet.data_reader import DataReader
from modules_actdet.data_writer import DataWriter
from os.path import join 
import os 
import cv2

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

# Place your downloaded cfg and weights under "checkpoints/"
YOLO_CONFIG = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/cfg'
YOLO_MODEL = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/cfg/yolo.cfg'
YOLO_WEIGHTS = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/checkpoints/yolo/yolo.weights'

GPU_ID = 0
GPU_UTIL = 0.5
YOLO_THRES = 0.4
YOLO_PEOPLE_LABEL = 'person'

'''
Input: {'img':img_np_array, 'meta':{'frame_id':frame_id}}

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'label': label,
                        'conf': conf_score
                        }]
                }
        }
'''
class YOLO:
    # dets = []
    # input = {}

    @staticmethod
    def Setup():
        opt = { "config": YOLO_CONFIG,  
                "model": YOLO_MODEL, 
                "load": YOLO_WEIGHTS, 
                # "gpuName": GPU_ID,
                # "gpu": GPU_UTIL,
                "threshold": YOLO_THRES
            }
        YOLO.tfnet = TFNet(opt)
        # self.log('init')


    def PreProcess(self, request, istub):
        # self.input = input 
        self.request_input = str(tensor_util.MakeNdarray(request.inputs["client_input"]))
        self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
        self.istub = istub

    def Apply(self):
        # if self.input:
        #     ichannel = grpc.insecure_channel("localhost:8500")
        #     self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)
        #     self.dets = self.tfnet.return_predict(self.input['img'], self.istub)   
        self.dets = YOLO.tfnet.return_predict(self.image, self.istub)
        

    def PostProcess(self):
        output = ""
        for d in self.dets:
            if d['label'] != YOLO_PEOPLE_LABEL:
                continue
            output += "%s|%s|%s|%s|%s|%s-" % (str(d['topleft']['x']), str(d['topleft']['y']), str(d['bottomright']['x']), str(d['bottomright']['y']), str(d['confidence']), str(d['label']))

        output = output[:-1]
        
        next_request = predict_pb2.PredictRequest()
        next_request.inputs['client_input'].CopyFrom(
          tf.make_tensor_proto(self.request_input))
        next_request.inputs['objdet_output'].CopyFrom(
          tf.make_tensor_proto(output))

        return next_request


        # output = self.input
        # if not self.input:
        #     return output 

        # output['meta']['obj'] = []
        # for d in self.dets:
        #     if d['label'] != YOLO_PEOPLE_LABEL:
        #         continue 
        #     output['meta']['obj'].append({'box':[int(d['topleft']['x']), int(d['topleft']['y']),
        #                                         int(d['bottomright']['x']), int(d['bottomright']['y'])],
        #                                         'label': d['label'],
        #                                         'conf': d['confidence']})
        # return output


#     def log(self, s):
#         print('[YOLO] %s' % s)


# ''' UNIT TEST '''
# if __name__ == '__main__':
#     yolo = YOLO()
#     yolo.Setup()

#     dr = DataReader()
#     dr.Setup('test/video.mp4')

#     dw = DataWriter()
#     dw.Setup('obj_det_res.npy')

#     cur_time = time()
#     cnt = 0 
#     while True:
#         d = dr.PostProcess()
#         print(cnt)
#         if not d:
#             break 
#         yolo.PreProcess(d)
#         yolo.Apply()
#         objs = yolo.PostProcess()
#         dw.PreProcess(objs['meta'])
#         cnt += 1

#     print('FPS: %.1f' % (float(cnt) / float(time() - cur_time)))
    
#     dw.save()
#     print('done')
