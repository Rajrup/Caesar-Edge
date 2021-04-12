import sys
import numpy as np
import tensorflow as tf
import os
import json
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util

SSD_THRES = 0.4
SSD_PEOPLE_LABEL = 1

class SSD:
    rclasses = []
    rbboxes = []
    rscores = []
    input = {}

    def Setup(self):
        self.output_dict = {}
        self.log('init done')

    def PreProcess(self, input):
        self.input = input
        self.input_image = np.expand_dims(self.input['img'], 0)

    def Apply(self):

        ichannel = grpc.insecure_channel("localhost:8500")
        self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

        self.internal_request = predict_pb2.PredictRequest()
        self.internal_request.model_spec.name = 'ssd_mobilenet_v1_coco_2017_11_17'
        self.internal_request.model_spec.signature_name = 'predict_images'

        self.internal_request.inputs['input_image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(self.input_image, shape=self.input_image.shape))

        # Run inference
        self.internal_result = self.istub.Predict(self.internal_request, 10.0) # 10 sec timeout

        self.output_dict['num_detections'] = tensor_util.MakeNdarray(self.internal_result.outputs['num_detections'])
        self.output_dict['detection_classes'] = tensor_util.MakeNdarray(self.internal_result.outputs['detection_classes'])
        self.output_dict['detection_boxes'] = tensor_util.MakeNdarray(self.internal_result.outputs['detection_boxes'])
        self.output_dict['detection_scores'] = tensor_util.MakeNdarray(self.internal_result.outputs['detection_scores'])

        # all outputs are float32 numpy arrays, so convert types as appropriate
        self.num_boxes = int(self.output_dict['num_detections'][0])
        self.rclasses = self.output_dict['detection_classes'][0].astype(np.uint8)
        self.rbboxes = self.output_dict['detection_boxes'][0]
        self.rscores = self.output_dict['detection_scores'][0]

    def PostProcess(self):
        output = self.input
        if not self.input:
            return output

        output['meta']['obj'] = []
        shape = self.input['img'].shape

        for i in range(self.rbboxes.shape[0]):
            if self.rclasses[i] != SSD_PEOPLE_LABEL:
                continue
            if self.rscores[i] < SSD_THRES:
                continue 
            bbox = self.rbboxes[i,:]
            p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            output['meta']['obj'].append(
                                            {'box':[p1[0],p1[1],p2[0],p2[1]], 
                                            'conf': self.rscores[i], 
                                            'label':self.rclasses[i]}
                                        )
        return output

    def log(self, s):
        print('[SSD] %s' % s)
