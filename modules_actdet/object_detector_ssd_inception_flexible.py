import cv2
import numpy as np
import os

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2

import string_int_label_map_pb2 as labelmap
from google.protobuf import text_format

INCEPTION_THRES = 0.4
INCEPTION_PEOPLE_LABEL = "person"

class ActDetInception:

  @staticmethod
  def Setup():
    s = open('%s/modules_traffic/mscoco_complete_label_map.pbtxt' % os.environ['TRAFFIC_JAMMER_PATH'], 'r').read()
    mymap = labelmap.StringIntLabelMap()
    ActDetInception._label_map = text_format.Parse(s, mymap)

  def decode_image_opencv(self, image, max_height = 800, swapRB = True, imagenet_mean = (0, 0, 0)):
    # image = cv2.imread(img_path, 1)
    (h, w) = image.shape[:2]
    image = self.image_resize(image, height = max_height)
    org  = image
    image = cv2.dnn.blobFromImage(image, scalefactor=1.0, mean = imagenet_mean, swapRB = swapRB)
    image = np.transpose(image, (0, 2, 3, 1))
    return image, org

  def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

  def box_normal_to_pixel(self, box, dim, scalefactor = 1):
    # https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
    height, width = dim[0], dim[1]
    ymin = int(box[0] * height * scalefactor)
    xmin = int(box[1] * width * scalefactor)

    ymax = int(box[2] * height * scalefactor)
    xmax= int(box[3] * width * scalefactor)
    return np.array([xmin, ymin, xmax, ymax])

  def get_label(self, index):
    return ActDetInception._label_map.item[index].display_name

  def PreProcess(self, request_input, istub, grpc_flag):
    if (grpc_flag):
      self.request_input = str(tensor_util.MakeNdarray(request_input.inputs["client_input"]))
      self.image = cv2.imdecode(np.fromstring(self.request_input, dtype = np.uint8), -1)
    else:
      self.image = request_input['client_input']

    self.input, self.org = self.decode_image_opencv(self.image)
    self.input = self.input.astype(np.uint8)

    self.istub = istub

  def Apply(self):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'actdet_inception'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['inputs'].CopyFrom(
      tf.contrib.util.make_tensor_proto(self.input, shape=self.input.shape))
    
    result = self.istub.Predict(request, 10.0)

    boxes = tf.make_ndarray(result.outputs['detection_boxes'])
    scores = tf.make_ndarray(result.outputs['detection_scores'])
    labels = tf.make_ndarray(result.outputs['detection_classes'])

    output = ""
    _draw = self.org.copy()
    # print(_draw.shape)
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score < INCEPTION_THRES:
          break
      dim = _draw.shape
      box = self.box_normal_to_pixel(box, dim)
      b = box.astype(int)
      class_label = self.get_label(int(label))
      if (class_label == INCEPTION_PEOPLE_LABEL):
        # print("Label = %s at %s with score of %s" % (class_label, b, score))
        output += "%s|%s|%s|%s|%s|%s-" % (str(b[0]), str(b[1]), str(b[2]), str(b[3]), str(score), str(class_label))

    self.output = output[:-1]

  def PostProcess(self, grpc_flag):
    if (grpc_flag):
      pass
    else:
      result = dict()
      result['client_input'] = self.image
      result['objdet_output'] = self.output
      return result