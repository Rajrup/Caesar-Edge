import modules_actdet.acam.action_detector_serving as act
import sys 
import numpy as np 
from os.path import join 
import os 
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

# These two numbers should be same as your video input 
VIDEO_WID = 1280    
VIDEO_HEI = 720

CACHE_SIZE = 32      # number of consecutive frames 
MIN_TUBE_LEN = 16    # output a list of tube images every MIN_TUBE_LEN new frames

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
    input = {}
    probs = []

    def Setup(self):
        self.act_detector = act.Action_Detector('soft_attn')
        # self.updated_frames, self.temporal_rois, self.temporal_roi_batch_indices, cropped_frames = \
        #             self.act_detector.crop_tubes_in_tf_with_memory([CACHE_SIZE,
        #                                                             VIDEO_HEI,VIDEO_WID,3], 
        #                                                             CACHE_SIZE - MIN_TUBE_LEN)

        # self.rois, self.roi_batch_indices, self.pred_probs = \
        #             self.act_detector.define_inference_with_placeholders_noinput(cropped_frames)

        # self.act_detector.restore_model(ACAM_MODEL)


    def PreProcess(self, input):
        self.input = input 


    def Apply(self):
        if not self.input:
            return 

        obj = self.input['meta']['obj']
        tube_num = len(obj['actor_boxes'])

        frames = obj['frames']
        temporal_rois = obj['temporal_rois']
        nptmp1 = np.zeros(tube_num)
        norm_rois = obj['norm_rois']
        nptmp2 = np.arange(tube_num)

        ichannel = grpc.insecure_channel("localhost:8500")
        self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

        self.internal_request = predict_pb2.PredictRequest()
        self.internal_request.model_spec.name = 'actdet_acam'
        self.internal_request.model_spec.signature_name = 'predict_images'

        self.internal_request.inputs['updated_frames'].CopyFrom(
            tf.contrib.util.make_tensor_proto(frames, dtype = tf.float32, shape = frames.shape))

        self.internal_request.inputs['temporal_rois'].CopyFrom(
            tf.contrib.util.make_tensor_proto(temporal_rois, dtype = tf.float32, shape = temporal_rois.shape))

        self.internal_request.inputs['temporal_roi_batch_indices'].CopyFrom(
            tf.contrib.util.make_tensor_proto(nptmp1, dtype = tf.int32, shape = nptmp1.shape))

        self.internal_request.inputs['rois'].CopyFrom(
            tf.contrib.util.make_tensor_proto(norm_rois, dtype = tf.float32, shape = norm_rois.shape))

        self.internal_request.inputs['roi_batch_indices'].CopyFrom(
            tf.contrib.util.make_tensor_proto(nptmp2, dtype = tf.int32, shape = nptmp2.shape))

        self.internal_result = self.istub.Predict(self.internal_request, 10.0)
        self.probs = tensor_util.MakeNdarray(self.internal_result.outputs['output'])

        # if True:
        #     # Yitao-TLS-Begin
        #     # init_op = tf.global_variables_initializer()
        #     # self.act_detector.session.run(init_op)
        #     # with tf.variable_scope('ActionDetector'):
        #     with self.act_detector.act_graph.as_default():
        #       # init_op = tf.global_variables_initializer()
        #       # self.act_detector.session.run(init_op)

        #       export_path_base = "actdet_acam"
        #       export_path = os.path.join(
        #           compat.as_bytes(export_path_base),
        #           compat.as_bytes(str(FLAGS.model_version)))
        #       print('Exporting trained model to ', export_path)
        #       builder = saved_model_builder.SavedModelBuilder(export_path)

        #       tensor_info_x_updated_frames =              tf.saved_model.utils.build_tensor_info(self.updated_frames)
        #       tensor_info_x_temporal_rois =               tf.saved_model.utils.build_tensor_info(self.temporal_rois)
        #       tensor_info_x_temporal_roi_batch_indices =  tf.saved_model.utils.build_tensor_info(self.temporal_roi_batch_indices)
        #       tensor_info_x_rois =                        tf.saved_model.utils.build_tensor_info(self.rois)
        #       tensor_info_x_roi_batch_indices =           tf.saved_model.utils.build_tensor_info(self.roi_batch_indices)

        #       tensor_info_y = tf.saved_model.utils.build_tensor_info(self.pred_probs)

        #       prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        #           inputs={  'updated_frames':             tensor_info_x_updated_frames,
        #                     'temporal_rois':              tensor_info_x_temporal_rois,
        #                     'temporal_roi_batch_indices': tensor_info_x_temporal_roi_batch_indices,
        #                     'rois':                       tensor_info_x_rois,
        #                     'roi_batch_indices':          tensor_info_x_roi_batch_indices,
        #                   },
        #           outputs={'output': tensor_info_y},
        #           method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        #       legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        #       builder.add_meta_graph_and_variables(
        #           self.act_detector.session, [tf.saved_model.tag_constants.SERVING],
        #           signature_def_map={
        #               'predict_images':
        #                   prediction_signature,
        #           },
        #           legacy_init_op=legacy_init_op)

        #       builder.save()

        #       print('Done exporting!')
        #     # Yitao-TLS-End 

        
    def PostProcess(self):
        output = self.input 
        if not self.input or not len(self.probs):
            return {}

        output['meta']['obj'] = self.input['meta']['obj']['actor_boxes']
        for i in xrange(len(output['meta']['obj'])):
            act_probs = self.probs[i]
            order = np.argsort(act_probs)[::-1]
            # cur_actor_id = output['meta']['obj'][i]['tid']
            cur_results = []
            for pp in range(PRINT_TOP_K):
                cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
            output['meta']['obj'][i]['act'] = cur_results

        return output


    def log(self, s):
        print('[ACAM] %s' % s)


''' UNIT TEST '''
if __name__ == '__main__':    
    from modules.tube_manager import TubeManager
    tm = TubeManager()
    tm.Setup()

    acam = ACAM()
    acam.Setup()

    dr = DataReader()
    dr.Setup('test/video.mp4', 'track_res.npy')

    dw = DataWriter()
    dw.Setup('act_res.npy')

    cur_time = time()
    cnt = 0 
    while True:
        d = dr.PostProcess()
        print(cnt)
        if not d:
            break 
        tm.PreProcess(d)
        tm.Apply()
        tubes = tm.PostProcess()
        if tubes:
            acam.PreProcess(tubes)
            acam.Apply()
            res = acam.PostProcess()
            dw.PreProcess(res['meta'])
        cnt += 1

    print('FPS: %.1f' % (float(cnt) / float(time() - cur_time)))
    
    dw.save()
    print('done')