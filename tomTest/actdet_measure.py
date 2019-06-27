import time
import os
import pickle
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import threading
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/')

from modules_actdet.data_reader import DataReader
from modules_actdet.object_detector_ssd_flexible import SSD
from modules_actdet.object_detector_yolo_flexible import YOLO
from modules_actdet.reid_extractor_flexible import FeatureExtractor
from modules_actdet.tracker_deepsort_flexible import DeepSort
from modules_actdet.tube_manager_flexible import TubeManager
from modules_actdet.action_detector_acam_flexible import ACAM

from deep_sort.tracker import Tracker
from deep_sort import nn_matching 
from modules_actdet.acam.manage_tube import TManager

# ============ Video Input Modules ============
reader = DataReader()
reader.Setup("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/indoor_two_ppl.avi")

# ============ Object Detection Modules ============
ssd = SSD()
ssd.Setup()

yolo = YOLO()
yolo.Setup()

object_detector = ssd

# # ============ Tracking Modules ============
feature_extractor = FeatureExtractor()
feature_extractor.Setup()

deepsort = DeepSort()
deepsort.Setup()

tracker = deepsort

# ============ Action Detection Modules ============
tube_manager = TubeManager()
tube_manager.Setup()

acam = ACAM()
acam.Setup()

action_detector = acam

def findPreviousModule(route_table, measure_module):
  tmp = route_table.split('-')
  for i in range(len(tmp)):
    if (tmp[i] == measure_module):
      return tmp[i - 1]


ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
my_tracker = Tracker(metric, max_iou_distance=0.7, max_age=200, n_init=4)

my_tube_manager = TManager(cache_size=TubeManager.cache_size, min_tube_len=TubeManager.action_freq)

my_lock = threading.Lock()


simple_route_table = "SSD-FeatureExtractor-DeepSort-TubeManager-ACAM"
measure_module = "ACAM"
route_table = simple_route_table

sess_id = "chain_actdet-000"
frame_id = 0

pickle_directory = "/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % measure_module
if not os.path.exists(pickle_directory):
  os.makedirs(pickle_directory)


while (frame_id < 120):
  start = time.time()

  # get input
  if (measure_module == "SSD"):
    frame_info = "%s-%s" % (sess_id, frame_id)
    route_index = 0
    frame_data = reader.PostProcess()
    request_input = dict()
    request_input['client_input'] = frame_data['img']
    request_input['frame_info'] = frame_info
    request_input['route_table'] = route_table
    request_input['route_index'] = route_index
  elif (measure_module == "YOLO"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(frame_id).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "FeatureExtractor"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(frame_id).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "DeepSort"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(frame_id).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "TubeManager"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(frame_id).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "ACAM"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(frame_id).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)

  if (measure_module == "SSD"):
    module_instance = ssd
  elif (measure_module == "YOLO"):
    module_instance = yolo
  elif (measure_module == "FeatureExtractor"):
    module_instance = feature_extractor
  elif (measure_module == "DeepSort"):
    module_instance = deepsort
  elif (measure_module == "TubeManager"):
    module_instance = tube_manager
  elif (measure_module == "ACAM"):
    module_instance = action_detector

  if (measure_module == "DeepSort"):
    module_instance.PreProcess(request_input = request_input, istub = istub, tracker = my_tracker, my_lock = my_lock, grpc_flag = False)
  elif (measure_module == "TubeManager"):
    module_instance.PreProcess(request_input = request_input, istub = istub, tube_manager = my_tube_manager, my_lock = my_lock, grpc_flag = False)
  else:
    module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
  module_instance.Apply()
  next_request = module_instance.PostProcess(grpc_flag = False)

  next_request['frame_info'] = request_input['frame_info']
  next_request['route_table'] = request_input['route_table']
  next_request['route_index'] = request_input['route_index'] + 1

  end = time.time()
  print("duration = %s" % (end - start))

  if (measure_module == "SSD"):
    print(next_request["objdet_output"])
  elif (measure_module == "YOLO"):
    print(next_request["objdet_output"])
  elif (measure_module == "FeatureExtractor"):
    print(next_request["reid_output"])
  elif (measure_module == "DeepSort"):
    print(next_request["deepsort_output"])
  elif (measure_module == "TubeManager"):
    pass
    # print(next_request["temporal_rois_output"])
    # print(next_request["norm_rois_output"])
    # print(next_request["actor_boxes_output"])
  elif (measure_module == "ACAM"):
    if (next_request["FINAL"] != "None@None"):
      print(next_request["FINAL"])

  # pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
  # with open(pickle_output, 'w') as f:
  #   pickle.dump(next_request, f)

  frame_id += 1
