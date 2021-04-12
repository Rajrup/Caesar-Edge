import sys
import numpy as np
import tensorflow as tf
import os
import json

# Place your downloaded ckpt under "checkpoints/"
SSD_MODEL = './checkpoints/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'

SSD_THRES = 0.4
SSD_PEOPLE_LABEL = 1

class SSD:
    rclasses = []
    rbboxes = []
    rscores = []
    input = {}

    def Setup(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.sess = tf.Session(config=gpu_config)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(SSD_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.session = tf.Session()
        with tf.gfile.GFile(SSD_MODEL, "rb") as file_handle:
            od_graph_def = tf.GraphDef()
            od_graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(od_graph_def, name='')
        ops = tf.get_default_graph().get_operations()
        
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        self.image = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        self.log('init done')

    def PreProcess(self, input):
        self.input = input
        self.input_image = np.expand_dims(self.input['img'], 0)

    def Apply(self):

        # Run inference
        self.output_dict = self.sess.run(self.tensor_dict,
                                feed_dict={self.image: self.input_image})

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
