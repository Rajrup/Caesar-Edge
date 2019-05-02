from modules_actdet.acam.manage_tube import TManager
import sys 
import numpy as np 
from os.path import join 
import os
import pickle
import cv2
from time import time 

import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2

# # These two numbers should be same as your video input 
# VIDEO_WID = 1280    
# VIDEO_HEI = 720

# CACHE_SIZE = 32      # number of consecutive frames 
# MIN_TUBE_LEN = 16    # output a list of tube images every MIN_TUBE_LEN new frames

'''
Input: {'img': img_np_array, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'tid': track_id
                        }]
                }
        }

Output: {'img': None, 
        'meta':{
                'frame_id': frame_id, 
                'obj':{
                        'frames': list of cv2 frames, 
                        'temporal_rois': list of boxes, 
                        'norm_rois': list of boxes,
                        'tube_boxes': list of boxes
                        }
                }
        }
'''

class TubeManager:

    @staticmethod
    def Setup():
        # self.tmanager = TManager(cache_size=CACHE_SIZE, min_tube_len=MIN_TUBE_LEN)
        TubeManager.cache_size = 32            # every result requires 32 frames
        TubeManager.action_freq = 16           # update the act result every 16 frames 
        # TubeManager.print_top_k = 5            # show top 5 actions for each tube 

    def getTubeInput(self, request_input, frame_id, deepsort_output):
      print("[Yitao] @@@ debug @@@, deepsort_output = %s" % deepsort_output)
      output = {'img': request_input, 'meta': {'frame_id': frame_id}}
      output['meta']['obj'] = []
      if (len(deepsort_output) == 0):
          return output
      else:
          for tmp in deepsort_output.split('-'):
              tt = tmp.split('|')
              output['meta']['obj'].append({'box':[int(tt[0]), int(tt[1]), int(tt[2]), int(tt[3])], 'tid': int(tt[4])})
          return output

    def PreProcess(self, request, istub, tube_manager):
        self.istub = istub
        self.tube_manager = tube_manager
        self.request_input = str(tensor_util.MakeNdarray(request.inputs["client_input"]))
        self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
        deepsort_output = str(tensor_util.MakeNdarray(request.inputs["deepsort_output"]))
        self.frame_id = int(str(tensor_util.MakeNdarray(request.inputs["frame_info"])).split('-')[-1])

        tube_input = self.getTubeInput(self.image, self.frame_id, deepsort_output)
        self.tube_manager.add_frame(tube_input)

        # self.input = input 
        # if input:
        #     self.tmanager.add_frame(input)

        self.can_output = False

    def Apply(self):
        if (self.frame_id % TubeManager.action_freq != 0 or self.frame_id < TubeManager.cache_size):
            return
        elif (not self.tube_manager.has_new_tube()):
            return
        else:
            self.frames, self.temporal_rois, self.norm_rois, self.actor_boxes = self.tube_manager.new_tube_data()
            self.can_output = True
            return
        
    def PostProcess(self):
        if not self.can_output:
            frames_output = "None"
            temporal_rois_output = "None"
            norm_rois_output = "None"
            actor_boxes_output = "None"
        else:
            frames_output = pickle.dumps(self.frames)
            temporal_rois_output = pickle.dumps(self.temporal_rois)
            norm_rois_output = pickle.dumps(self.norm_rois)
            actor_boxes_output = pickle.dumps(self.actor_boxes)

        next_request = predict_pb2.PredictRequest()
        next_request.inputs['frames_output'].CopyFrom(
          tf.make_tensor_proto(frames_output))
        next_request.inputs['temporal_rois_output'].CopyFrom(
          tf.make_tensor_proto(temporal_rois_output))
        next_request.inputs['norm_rois_output'].CopyFrom(
          tf.make_tensor_proto(norm_rois_output))
        next_request.inputs['actor_boxes_output'].CopyFrom(
          tf.make_tensor_proto(actor_boxes_output)) 

        return next_request


    # def log(self, s):
    #     print('[TM] %s' % s)