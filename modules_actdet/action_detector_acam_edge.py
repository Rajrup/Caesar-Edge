import modules_actdet.acam.action_detector_serving as act
import sys 
import numpy as np 
from os.path import join 
import os 
import pickle
from time import time 
from modules_actdet.data_reader import DataReader
from modules_actdet.data_writer import DataWriter

# # Yitao-TLS-Begin
import tensorflow as tf
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
# ACAM_MODEL = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/checkpoints/acam/model_ckpt_soft_attn_pooled_cosine_drop_ava-130'

# # These two numbers should be same as your video input 
# VIDEO_WID = 1280    
# VIDEO_HEI = 720

# CACHE_SIZE = 32      # number of consecutive frames 
# MIN_TUBE_LEN = 16    # output a list of tube images every MIN_TUBE_LEN new frames

PRINT_TOP_K = 5      # show the top 5 possible action for current second 

'''
Input: {'img': img_np_array, 
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

Output: {'img': None, 
        'meta':{
                'frame_id': frame_id, 
                'obj':[{
                        'box': [x0,y0,x1,y1],
                        'tid': track_id,
                        'act': [(act_label, act_prob)]
                        }]
                }
        }
'''
class ACAM:
    # input = {}
    # probs = []

    @staticmethod
    def Setup():
        # self.act_detector = act.Action_Detector('soft_attn')
        # self.updated_frames, self.temporal_rois, self.temporal_roi_batch_indices, cropped_frames = \
        #             self.act_detector.crop_tubes_in_tf_with_memory([CACHE_SIZE,
        #                                                             VIDEO_HEI,VIDEO_WID,3], 
        #                                                             CACHE_SIZE - MIN_TUBE_LEN)

        # self.rois, self.roi_batch_indices, self.pred_probs = \
        #             self.act_detector.define_inference_with_placeholders_noinput(cropped_frames)

        # self.act_detector.restore_model(ACAM_MODEL)
        pass

    def PreProcess(self, request, istub):
        # self.input = input 
        self.istub = istub
        self.has_input = False
        if (str(tensor_util.MakeNdarray(request.inputs["frames_output"])) == "None"):
            return
        else:
            self.has_input = True
            self.frames = pickle.loads(str(tensor_util.MakeNdarray(request.inputs["frames_output"])))
            self.temporal_rois = pickle.loads(str(tensor_util.MakeNdarray(request.inputs["temporal_rois_output"])))
            self.norm_rois = pickle.loads(str(tensor_util.MakeNdarray(request.inputs["norm_rois_output"])))
            self.actor_boxes = pickle.loads(str(tensor_util.MakeNdarray(request.inputs["actor_boxes_output"])))


    def Apply(self):
        if (not self.has_input):
            self.probs = []
        else:
            self.internal_request = predict_pb2.PredictRequest()
            self.internal_request.model_spec.name = 'actdet_acam'
            self.internal_request.model_spec.signature_name = 'predict_images'

            tube_num = len(self.actor_boxes)
            nptmp1 = np.zeros(tube_num)
            nptmp2 = np.arange(tube_num)

            self.internal_request.inputs['updated_frames'].CopyFrom(
                tf.contrib.util.make_tensor_proto(self.frames, dtype = tf.float32, shape = self.frames.shape))

            self.internal_request.inputs['temporal_rois'].CopyFrom(
                tf.contrib.util.make_tensor_proto(self.temporal_rois, dtype = tf.float32, shape = self.temporal_rois.shape))

            self.internal_request.inputs['temporal_roi_batch_indices'].CopyFrom(
                tf.contrib.util.make_tensor_proto(nptmp1, dtype = tf.int32, shape = nptmp1.shape))

            self.internal_request.inputs['rois'].CopyFrom(
                tf.contrib.util.make_tensor_proto(self.norm_rois, dtype = tf.float32, shape = self.norm_rois.shape))

            self.internal_request.inputs['roi_batch_indices'].CopyFrom(
                tf.contrib.util.make_tensor_proto(nptmp2, dtype = tf.int32, shape = nptmp2.shape))

            self.internal_result = self.istub.Predict(self.internal_request, 10.0)
            self.probs = tensor_util.MakeNdarray(self.internal_result.outputs['output'])
        
    def PostProcess(self):
        if (not len(self.probs)):
            abstr = "None"
            resstr = "None"
        else:
            abstr = ""
            for ab in self.actor_boxes:
                abstr += "%d|%d|%d|%d|%d-" % (ab['box'][0][0], ab['box'][0][1], ab['box'][0][2], ab['box'][0][3], ab['tid'])
            abstr = abstr[:-1]
            resstr = ""
            for i in xrange(len(self.actor_boxes)):
                act_probs = self.probs[i]
                order = np.argsort(act_probs)[::-1]
                for pp in range(PRINT_TOP_K):
                    resstr += "%s|%s|" % (str(act.ACTION_STRINGS[order[pp]]), str(act_probs[order[pp]]))
                resstr = resstr[:-1]
                resstr += '-'
            resstr = resstr[:-1]

        result = "%s@%s" % (abstr, resstr)

        next_request = predict_pb2.PredictRequest()
        next_request.inputs['FINAL'].CopyFrom(
          tf.make_tensor_proto(result))

        return next_request