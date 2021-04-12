import sys
import numpy as np
import tensorflow as tf
import os
import json

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("./modules_actdet/models/research")

# Place your downloaded ckpt under "checkpoints/"
SSD_MODEL = './checkpoints/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './modules_actdet/research/object_detection/data/mscoco_label_map.pbtxt'

# Config File for the Resnet Model
CONFIG_FILE = "./cfg/config.json"

DS_HOME = './modules_actdet/reid'
sys.path.append(DS_HOME)

import reid_nets.resnet_v1_50 as model
import reid_heads.fc1024 as head
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
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        self.log('init done ')

    def log(self, s):
        print('[SSD] %s' % s)

class REID:
    def Setup(self):
        config = json.loads(open(CONFIG_FILE, 'r').read())

        gpu_options = tf.GPUOptions(allow_growth=True)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=gpu_config)

        # self.sess = tf.Session(config=gpu_config)
        net_input_size = (config['input_height'], config['input_width'])
        checkpoint_filename = os.path.join(config['experiment_root'], config['checkpoint_filename'])

        self.image = tf.placeholder(tf.float32, (None, net_input_size[0], net_input_size[1], 3))

        self.endpoints, _ = model.endpoints(self.image, is_training=False)
        with tf.name_scope('head'):
            self.endpoints = head.head(self.endpoints, config['embedding_dim'], is_training=False)

        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_filename)

        self.feature_dim = config['embedding_dim']
        self.image_shape = [config['input_height'], config['input_width']]

        self.log('init done')

    def log(self, s):
        print('[REID] %s' % s)

def main():
# ============ Object Detection Modules ============
    ssd = SSD()
    ssd.Setup()

# ============ Tracking Modules ============
    feature_extractor = REID()
    feature_extractor.Setup()

    print("Done")

if __name__ == '__main__':
    main()