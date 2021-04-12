import sys
import numpy as np
import tensorflow as tf
import os
import json

# Yitao-TLS-Begin
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

# TF Serving flags
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("./modules_actdet/models/research")

# Place your downloaded ckpt under "checkpoints/"
SSD_MODEL = './checkpoints/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

# Path to tf servable model
EXPORT_SSD_MODEL = './tf_servable/ssd_mobilenet_v1_coco_2017_11_17'
EXPORT_REID_MODEL = './tf_servable/triplet-reid'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './modules_actdet/research/object_detection/data/mscoco_label_map.pbtxt'

# Config File for the Resnet based REID Model
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
        self.tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        self.image = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        self.log('init done')

    def export_model(self):
        # Yitao-TLS-Begin
        if not os.path.exists(EXPORT_SSD_MODEL):
            os.makedirs(EXPORT_SSD_MODEL)

        export_path = os.path.join(
            compat.as_bytes(os.path.abspath(EXPORT_SSD_MODEL)),
            compat.as_bytes(str(FLAGS.model_version)))

        self.log('Exporting trained model to {}'.format(export_path))
        builder = saved_model_builder.SavedModelBuilder(export_path)

        input_image = tf.saved_model.utils.build_tensor_info(self.image)
        output_num_detections = tf.saved_model.utils.build_tensor_info(self.tensor_dict['num_detections'])
        output_detection_boxes = tf.saved_model.utils.build_tensor_info(self.tensor_dict['detection_boxes'])
        output_detection_scores = tf.saved_model.utils.build_tensor_info(self.tensor_dict['detection_scores'])
        output_detection_classes = tf.saved_model.utils.build_tensor_info(self.tensor_dict['detection_classes'])

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs = {'input_image': input_image},
            outputs = { 'num_detections': output_num_detections, 
                        'detection_boxes': output_detection_boxes,
                        'detection_scores': output_detection_scores,
                        'detection_classes': output_detection_classes},
            method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()

        self.log('Exporting Done!')
        # Yitao-TLS-End

    def log(self, s):
        print('[SSD] %s' % s)

class REID:
    def Setup(self):
        config = json.loads(open(CONFIG_FILE, 'r').read())

        gpu_options = tf.GPUOptions(allow_growth=True)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.sess = tf.Session(config=gpu_config)

        # self.session = tf.Session()
        self.image_shape = [config['input_height'], config['input_width'], config['input_channel']]
        net_input_size = (self.image_shape[0], self.image_shape[1])
        checkpoint_filename = os.path.join(config['experiment_root'], config['checkpoint_filename'])

        self.image = tf.placeholder(tf.float32, (None, net_input_size[0], net_input_size[1], 3))

        self.endpoints, _ = model.endpoints(self.image, is_training=False)
        with tf.name_scope('head'):
            self.endpoints = head.head(self.endpoints, config['embedding_dim'], is_training=False)

        # self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_filename)

        self.feature_dim = config['embedding_dim']

        self.log('init done')

    def export_model(self):
        # Yitao-TLS-Begin
        if not os.path.exists(EXPORT_REID_MODEL):
            os.makedirs(EXPORT_REID_MODEL)

        export_path = os.path.join(
            compat.as_bytes(os.path.abspath(EXPORT_REID_MODEL)),
            compat.as_bytes(str(FLAGS.model_version)))

        self.log('Exporting trained model to {}'.format(export_path))
        builder = saved_model_builder.SavedModelBuilder(export_path)

        input_image = tf.saved_model.utils.build_tensor_info(self.image)
        output_embedding = tf.saved_model.utils.build_tensor_info(self.endpoints['emb'])

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs = {'input_image': input_image},
            outputs = { 'output_embedding': output_embedding},
            method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()

        self.log('Exporting Done!')
        # Yitao-TLS-End

    def log(self, s):
        print('[REID] %s' % s)

def main():
# ============ Object Detection Modules ============
    ssd = SSD()
    ssd.Setup()
    # ssd.export_model()

# ============ Tracking Modules ============
    feature_extractor = REID()
    feature_extractor.Setup()
    feature_extractor.export_model()

    print("Done")

if __name__ == '__main__':
    main()