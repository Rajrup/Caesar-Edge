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
        TubeManager.cache_size = 32            # every result requires 32 frames
        TubeManager.action_freq = 16           # update the act result every 16 frames 

    def getTubeInput(self, request_input, frame_id, deepsort_output):
      # print("[Yitao] @@@ debug @@@, deepsort_output = %s" % deepsort_output)
      output = {'img': request_input, 'meta': {'frame_id': frame_id}}
      output['meta']['obj'] = []
      if (len(deepsort_output) == 0):
          return output
      else:
          for tmp in deepsort_output.split('-'):
              tt = tmp.split('|')
              output['meta']['obj'].append({'box':[int(tt[0]), int(tt[1]), int(tt[2]), int(tt[3])], 'tid': int(tt[4])})
          return output

    def PreProcess(self, request_input, istub, tube_manager, my_lock, grpc_flag):
        self.istub = istub
        self.tube_manager = tube_manager
        self.my_lock = my_lock

        if (grpc_flag):
            self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
            self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
            deepsort_output = str(tensor_util.MakeNdarray(request_input.inputs["deepsort_output"]))
            self.frame_id = int(str(tensor_util.MakeNdarray(request_input.inputs["frame_info"])).split('-')[-1])
        else:
            self.image = request_input['client_input']
            deepsort_output = request_input['deepsort_output']
            self.frame_id = int(request_input['frame_info'].split('-')[-1])

        tube_input = self.getTubeInput(self.image, self.frame_id, deepsort_output)

        self.my_lock.acquire()
        self.tube_manager.add_frame(tube_input)
        self.my_lock.release()

        self.can_output = False

    def Apply(self):
        # if (self.frame_id % 32 != 0 or self.frame_id < TubeManager.cache_size):
        if (self.frame_id % TubeManager.action_freq != 0 or self.frame_id < TubeManager.cache_size):
            return
        elif (not self.tube_manager.has_new_tube()):
            return
        else:
            self.my_lock.acquire()
            self.frames, self.temporal_rois, self.norm_rois, self.actor_boxes = self.tube_manager.new_tube_data()
            self.my_lock.release()

            self.can_output = True
            return
        
    def PostProcess(self, grpc_flag):
        if (grpc_flag):
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
        else:
            if not self.can_output:
                frames_output = "None"
                temporal_rois_output = "None"
                norm_rois_output = "None"
                actor_boxes_output = "None"
            else:
                frames_output = self.frames
                temporal_rois_output = self.temporal_rois
                norm_rois_output = self.norm_rois
                actor_boxes_output = self.actor_boxes
                # print(frames_output.nbytes)
                # print(temporal_rois_output.nbytes)
                # print(norm_rois_output.nbytes)
                # print(sys.getsizeof(actor_boxes_output))
            result = dict()
            result['frames_output'] = frames_output
            result['temporal_rois_output'] = temporal_rois_output
            result['norm_rois_output'] = norm_rois_output
            result['actor_boxes_output'] = actor_boxes_output
            # print("[TubeManager] size of result = %s" % str(sys.getsizeof(result)))
            # print(sys.getsizeof(frames_output))
            return result
