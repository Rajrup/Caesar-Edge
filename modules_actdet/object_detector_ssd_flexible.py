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
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util
# # Yitao-TLS-End

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

    @staticmethod
    def Setup():
        SSD.thres = SSD_THRES  
        SSD.nms_thres = SSD_NMS
        SSD.net_shape = SSD_NET_SHAPE

        ssd_net = ssd_vgg_512.SSDNet()

        SSD.ssd_anchors = ssd_net.anchors(SSD.net_shape)
        SSD.total_classes = 21



    def PreProcess(self, request_input, istub, grpc_flag):
        if (grpc_flag):
            self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
            self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
        else:
            self.image = request_input['client_input']

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

        output = ""
        shape = self.image.shape

        for i in xrange(self.rbboxes.shape[0]):
            if self.rclasses[i] != SSD_PEOPLE_LABEL:
                continue
            bbox = self.rbboxes[i]
            p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            output += "%s|%s|%s|%s|%s|%s-" % (str(p1[0]), str(p1[1]), str(p2[0]), str(p2[1]), str(self.rscores[i]), str(self.rclasses[i]))

        self.output = output[:-1]

    def PostProcess(self, grpc_flag):
        if (grpc_flag):
            try:
                self.request_input
            except AttributeError:
                self.request_input = cv2.imencode('.jpg', self.image)[1].tostring()
                
            next_request = predict_pb2.PredictRequest()
            next_request.inputs['client_input'].CopyFrom(
              tf.make_tensor_proto(self.request_input))
            next_request.inputs['objdet_output'].CopyFrom(
              tf.make_tensor_proto(self.output))
            return next_request
        else:
            result = dict()
            result['client_input'] = self.image
            result['objdet_output'] = self.output
            return result
