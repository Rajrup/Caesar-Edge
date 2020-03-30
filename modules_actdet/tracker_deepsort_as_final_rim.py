import numpy as np 
from os.path import join 
import os 
import sys
import pickle
from time import time 
from modules_actdet.data_reader import DataReader
from modules_actdet.data_writer import DataWriter
import cv2

import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2

DS_HOME = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/modules_actdet/deep_sort'
sys.path.insert(0, DS_HOME)
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching 

'''
Input: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,w,h],
                        'conf': conf_score
                        'feature': feature_array
                        }]
                }
        }

Output: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'tid': track_id
                        }]
                }
        }
'''
class DeepSortFinal:

    @staticmethod
    def Setup():
        pass

    def PreProcess(self, request, istub, tracker, my_lock, grpc_flag):
        self.tracker = tracker
        self.my_lock = my_lock
        self.istub = istub

        if (grpc_flag):
            self.image = tensor_util.MakeNdarray(request.inputs["client_input"])
            self.objdet_output = str(tensor_util.MakeNdarray(request.inputs["objdet_output"]))
            self.features = pickle.loads(str(tensor_util.MakeNdarray(request.inputs["reid_output"])))
        else:
            self.image = request['client_input']
            self.objdet_output = request['objdet_output']
            self.features = request['reid_output']

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
        detection_list = [Detection(self.ds_boxes[i], self.scores[i], self.features[i]) for i in xrange(len(self.ds_boxes))]

        self.my_lock.acquire()
        self.tracker.predict()
        self.tracker.update(detection_list)
        self.my_lock.release()

        output = ""
        for tk in self.tracker.tracks:
            if not tk.is_confirmed() or tk.time_since_update > 1:
                continue
            left, top, width, height = map(int, tk.to_tlwh())
            track_id = tk.track_id
            output += "%s|%s|%s|%s|%s-" % (str(left), str(top), str(width), str(height), str(track_id))

        output = output[:-1]
        self.output = output.replace("--", "-") # weird bug...

    def PostProcess(self, grpc_flag):
        if (grpc_flag):
            next_request = predict_pb2.PredictRequest()
            # next_request.inputs['client_input'].CopyFrom(
            #   tf.make_tensor_proto(self.image))
            next_request.inputs['FINAL'].CopyFrom(
              tf.make_tensor_proto(self.output))
        else:
            next_request = dict()
            # next_request['client_input'] = self.image
            next_request['FINAL'] = self.output
            # print("[tracker] size of result = %s" % str(sys.getsizeof(result)))
        return next_request
