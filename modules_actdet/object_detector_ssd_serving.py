import sys
import numpy as np
import tensorflow as tf
from time import time 
from modules_actdet.data_reader import DataReader
from modules_actdet.data_writer import DataWriter
from os.path import join 
import os 

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
    rclasses = []
    rbboxes = []
    rscores = []
    input = {}

    def Setup(self):
        # self.ckpt = SSD_CKPT
        self.thres = SSD_THRES  
        self.nms_thres = SSD_NMS
        self.net_shape = SSD_NET_SHAPE

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
        self.ssd_anchors = ssd_net.anchors(self.net_shape)
        self.total_classes = 21

        # self.log('init done ')


    def PreProcess(self, input):
        self.input = input 


    def Apply(self):
        # if not self.input:
        #     return 

        # rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.pred, self.loc, self.bbx],
        #                                                         feed_dict={self.img_input: self.input['img']})

        # if False:
        #   # Yitao-TLS-Begin
        #   export_path_base = "actdet_ssd"
        #   export_path = os.path.join(
        #       compat.as_bytes(export_path_base),
        #       compat.as_bytes(str(FLAGS.model_version)))
        #   print('Exporting trained model to ', export_path)
        #   builder = saved_model_builder.SavedModelBuilder(export_path)

        #   tensor_info_x = tf.saved_model.utils.build_tensor_info(self.img_input)

        #   tensor_info_y10 = tf.saved_model.utils.build_tensor_info(self.pred[0])
        #   tensor_info_y11 = tf.saved_model.utils.build_tensor_info(self.pred[1])
        #   tensor_info_y12 = tf.saved_model.utils.build_tensor_info(self.pred[2])
        #   tensor_info_y13 = tf.saved_model.utils.build_tensor_info(self.pred[3])
        #   tensor_info_y14 = tf.saved_model.utils.build_tensor_info(self.pred[4])
        #   tensor_info_y15 = tf.saved_model.utils.build_tensor_info(self.pred[5])
        #   tensor_info_y16 = tf.saved_model.utils.build_tensor_info(self.pred[6])

        #   tensor_info_y20 = tf.saved_model.utils.build_tensor_info(self.loc[0])
        #   tensor_info_y21 = tf.saved_model.utils.build_tensor_info(self.loc[1])
        #   tensor_info_y22 = tf.saved_model.utils.build_tensor_info(self.loc[2])
        #   tensor_info_y23 = tf.saved_model.utils.build_tensor_info(self.loc[3])
        #   tensor_info_y24 = tf.saved_model.utils.build_tensor_info(self.loc[4])
        #   tensor_info_y25 = tf.saved_model.utils.build_tensor_info(self.loc[5])
        #   tensor_info_y26 = tf.saved_model.utils.build_tensor_info(self.loc[6])

        #   tensor_info_y3 = tf.saved_model.utils.build_tensor_info(self.bbx)

        #   prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        #       inputs={'input': tensor_info_x},
        #       outputs={ 'predictions0': tensor_info_y10,
        #                 'predictions1': tensor_info_y11,
        #                 'predictions2': tensor_info_y12,
        #                 'predictions3': tensor_info_y13,
        #                 'predictions4': tensor_info_y14,
        #                 'predictions5': tensor_info_y15,
        #                 'predictions6': tensor_info_y16,

        #                 'localisations0': tensor_info_y20,
        #                 'localisations1': tensor_info_y21,
        #                 'localisations2': tensor_info_y22,
        #                 'localisations3': tensor_info_y23,
        #                 'localisations4': tensor_info_y24,
        #                 'localisations5': tensor_info_y25,
        #                 'localisations6': tensor_info_y26,

        #                 'bbox_img': tensor_info_y3},
        #       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        #   legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        #   builder.add_meta_graph_and_variables(
        #       self.isess, [tf.saved_model.tag_constants.SERVING],
        #       signature_def_map={
        #           'predict_images':
        #               prediction_signature,
        #       },
        #       legacy_init_op=legacy_init_op)

        #   builder.save()

        #   print('Done exporting!')
        #   # Yitao-TLS-End

        ichannel = grpc.insecure_channel("localhost:8500")
        self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

        self.internal_request = predict_pb2.PredictRequest()
        self.internal_request.model_spec.name = 'actdet_ssd'
        self.internal_request.model_spec.signature_name = 'predict_images'
        self.internal_request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.input['img'], shape=self.input['img'].shape))

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
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=self.thres, img_shape=self.net_shape, num_classes=self.total_classes, decode=True)
    
        self.rbboxes = np_methods.bboxes_clip(rbbox_img, self.rbboxes)
        self.rclasses, self.rscores, self.rbboxes = np_methods.bboxes_sort(self.rclasses, self.rscores, 
                                                                        self.rbboxes, top_k=400)
        self.rclasses, self.rscores, self.rbboxes = np_methods.bboxes_nms(self.rclasses, self.rscores, 
                                                                        self.rbboxes, nms_threshold=self.nms_thres)
        self.rbboxes = np_methods.bboxes_resize(rbbox_img, self.rbboxes)


    def PostProcess(self):
        output = self.input
        if not self.input:
            return output

        output['meta']['obj'] = []
        shape = self.input['img'].shape
        for i in xrange(self.rbboxes.shape[0]):
            if self.rclasses[i] != SSD_PEOPLE_LABEL:
                continue 
            bbox = self.rbboxes[i]
            p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            output['meta']['obj'].append({'box':[p1[0],p1[1],p2[0],p2[1]], 'conf': self.rscores[i], 
                                                                            'label':self.rclasses[i]})
        return output 


    def log(self, s):
        print('[SSD] %s' % s)


''' UNIT TEST '''
if __name__ == '__main__':
    ssd = SSD()
    ssd.Setup()

    dr = DataReader()
    dr.Setup('test/video.mp4')

    dw = DataWriter()
    dw.Setup('obj_det_res.npy')

    cur_time = time()
    cnt = 0 
    while True:
        d = dr.PostProcess()
        print(cnt)
        if not d:
            break 
        ssd.PreProcess(d)
        ssd.Apply()
        objs = ssd.PostProcess()
        dw.PreProcess(objs['meta'])
        cnt += 1

    print('FPS: %.1f' % (float(cnt) / float(time() - cur_time)))
    
    dw.save()
    print('done')