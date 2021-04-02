import sys
import numpy as np
import tensorflow as tf
import os
import json

# Place your downloaded ckpt under "checkpoints/"
SSD_CKPT = './checkpoints/ssd/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'

SSD_HOME = './modules_actdet/SSD-Tensorflow'
sys.path.insert(0, SSD_HOME)
from nets import ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing

# Config File for the Resnet Model
CONFIG_FILE = "./cfg/config.json"

DS_HOME = './modules_actdet/deep_sort'
sys.path.insert(0, DS_HOME)

import reid_nets.resnet_v1_50 as model
import reid_heads.fc1024 as head

SSD_THRES = 0.4
SSD_NMS = 0.45
SSD_NET_SHAPE = (512, 512)
SSD_PEOPLE_LABEL = 15

class SSD:
    rclasses = []
    rbboxes = []
    rscores = []
    input = {}

    def Setup(self):
        self.ckpt = SSD_CKPT
        self.thres = SSD_THRES  
        self.nms_thres = SSD_NMS
        self.net_shape = SSD_NET_SHAPE

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        
        self.isess = tf.InteractiveSession(config=config)

        data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                                        self.img_input, None, None, self.net_shape, data_format, 
                                        resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        
        self.image_4d = tf.expand_dims(image_pre, 0)
        self.bbx = bbox_img

        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_512.SSDNet()
        slim = tf.contrib.slim
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            predictions, localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, self.ckpt)

        self.pred = predictions
        self.loc = localisations
        self.ssd_anchors = ssd_net.anchors(self.net_shape)
        self.total_classes = 21

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
        net_input_size = (config['net_input_height'], config['net_input_width'])
        checkpoint_filename = os.path.join(config['experiment_root'], config['checkpoint_filename'])

        self.image = tf.placeholder(tf.float32, (None, net_input_size[0], net_input_size[1], 3))

        self.endpoints, _ = model.endpoints(self.image, is_training=False)
        with tf.name_scope('head'):
            self.endpoints = head.head(self.endpoints, config['embedding_dim'], is_training=False)

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_filename)

        self.feature_dim = config['embedding_dim']
        self.image_shape = [config['net_input_height'], config['net_input_width']]

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