import sys
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

# # Place your downloaded ckpt under "checkpoints/"
# SSD_CKPT = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/checkpoints/ssd/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'

SSD_HOME = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/modules_actdet/SSD-Tensorflow'
sys.path.insert(0, SSD_HOME)
from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

SSD_THRES = 0.4
SSD_NMS = 0.45
SSD_NET_SHAPE = (512, 512)
SSD_PEOPLE_LABEL = 15

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
class SSD:
    # rclasses = []
    # rbboxes = []
    # rscores = []
    # input = {}

    @staticmethod
    def Setup():
        # self.ckpt = SSD_CKPT
        SSD.thres = SSD_THRES  
        SSD.nms_thres = SSD_NMS
        SSD.net_shape = SSD_NET_SHAPE

        # gpu_options = tf.GPUOptions(allow_growth=True)
        # config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        
        # self.isess = tf.InteractiveSession(config=config)

        # data_format = 'NHWC'
        # self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

        # image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        #                                 self.img_input, None, None, self.net_shape, data_format, 
        #                                 resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        
        # self.image_4d = tf.expand_dims(image_pre, 0)
        # self.bbx = bbox_img

        # reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_512.SSDNet()
        # slim = tf.contrib.slim
        # with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        #     predictions, localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

        # self.isess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # saver.restore(self.isess, self.ckpt)

        # self.pred = predictions
        # self.loc = localisations
        SSD.ssd_anchors = ssd_net.anchors(SSD.net_shape)
        SSD.total_classes = 21

        # self.log('init done ')


    def PreProcess(self, request, istub):
        # self.input = input
        self.request_input = str(tensor_util.MakeNdarray(request.inputs["client_input"]))
        self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
        self.istub = istub

        self.internal_request = predict_pb2.PredictRequest()
        self.internal_request.model_spec.name = "actdet_ssd"
        self.internal_request.model_spec.signature_name = 'predict_images'

        self.internal_request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.image, shape=self.image.shape)) 


    def Apply(self):

        self.internal_result = self.istub.Predict(self.internal_request, 10.0)

        rpredictions = [tensor_util.MakeNdarray(self.internal_result.outputs['predictions0']),
                        tensor_util.MakeNdarray(self.internal_result.outputs['predictions1']),
                        tensor_util.MakeNdarray(self.internal_result.outputs['predictions2']),
                        tensor_util.MakeNdarray(self.internal_result.outputs['predictions3']),
                        tensor_util.MakeNdarray(self.internal_result.outputs['predictions4']),
                        tensor_util.MakeNdarray(self.internal_result.outputs['predictions5']),
                        tensor_util.MakeNdarray(self.internal_result.outputs['predictions6'])]
        rlocalisations = [tensor_util.MakeNdarray(self.internal_result.outputs['localisations0']),
                          tensor_util.MakeNdarray(self.internal_result.outputs['localisations1']),
                          tensor_util.MakeNdarray(self.internal_result.outputs['localisations2']),
                          tensor_util.MakeNdarray(self.internal_result.outputs['localisations3']),
                          tensor_util.MakeNdarray(self.internal_result.outputs['localisations4']),
                          tensor_util.MakeNdarray(self.internal_result.outputs['localisations5']),
                          tensor_util.MakeNdarray(self.internal_result.outputs['localisations6'])]
        rbbox_img = tensor_util.MakeNdarray(self.internal_result.outputs['bbox_img'])









        self.rclasses, self.rscores, self.rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, SSD.ssd_anchors,
                select_threshold=SSD.thres, img_shape=SSD.net_shape, num_classes=SSD.total_classes, decode=True)
    
        self.rbboxes = np_methods.bboxes_clip(rbbox_img, self.rbboxes)
        self.rclasses, self.rscores, self.rbboxes = np_methods.bboxes_sort(self.rclasses, self.rscores, 
                                                                        self.rbboxes, top_k=400)
        self.rclasses, self.rscores, self.rbboxes = np_methods.bboxes_nms(self.rclasses, self.rscores, 
                                                                        self.rbboxes, nms_threshold=SSD.nms_thres)
        self.rbboxes = np_methods.bboxes_resize(rbbox_img, self.rbboxes)


    def PostProcess(self):
        output = ""
        shape = self.image.shape

        for i in xrange(self.rbboxes.shape[0]):
            if self.rclasses[i] != SSD_PEOPLE_LABEL:
                continue
            bbox = self.rbboxes[i]
            p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            output += "%s|%s|%s|%s|%s|%s-" % (str(p1[0]), str(p1[1]), str(p2[0]), str(p2[1]), str(self.rscores[i]), str(self.rclasses[i]))

        output = output[:-1]

        next_request = predict_pb2.PredictRequest()
        next_request.inputs['client_input'].CopyFrom(
          tf.make_tensor_proto(self.request_input))
        next_request.inputs['objdet_output'].CopyFrom(
          tf.make_tensor_proto(output))

        return next_request


#     def log(self, s):
#         print('[SSD] %s' % s)


# ''' UNIT TEST '''
# if __name__ == '__main__':
#     ssd = SSD()
#     ssd.Setup()

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
#         ssd.PreProcess(d)
#         ssd.Apply()
#         objs = ssd.PostProcess()
#         dw.PreProcess(objs['meta'])
#         cnt += 1

#     print('FPS: %.1f' % (float(cnt) / float(time() - cur_time)))
    
#     dw.save()
#     print('done')