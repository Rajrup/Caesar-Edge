import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import time
import threading
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/')

from modules_actdet.data_reader import DataReader
from modules_actdet.object_detector_ssd_flexible import SSD
from modules_actdet.object_detector_yolo_flexible import YOLO
from modules_actdet.object_detector_yolotiny_flexible import ActDetYoloTiny
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

yolotiny = ActDetYoloTiny()
yolotiny.Setup()

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




ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
my_tracker = Tracker(metric, max_iou_distance=0.7, max_age=200, n_init=4)

my_tube_manager = TManager(cache_size=TubeManager.cache_size, min_tube_len=TubeManager.action_freq)

my_lock = threading.Lock()


# simple_route_table = "SSD-FeatureExtractor-DeepSort"
simple_route_table = "SSD-FeatureExtractor-DeepSort-TubeManager-ACAM"
route_table = simple_route_table

sess_id = "chain_actdet-000"
frame_id = 0

while (frame_id < 160):
# while (True):
  start = time.time()
  
  frame_info = "%s-%s" % (sess_id, frame_id)

  route_index = 0

  frame_data = reader.PostProcess()
  if not frame_data:  # end of video 
    break
  
  request_input = dict()
  request_input['client_input'] = frame_data['img']
  request_input['frame_info'] = frame_info
  request_input['route_table'] = route_table
  request_input['route_index'] = route_index

  for i in range(len(route_table.split('-'))):
    current_model = route_table.split('-')[request_input['route_index']]

    if (current_model == "SSD"):
      module_instance = ssd
    elif (current_model == "YOLO"):
      module_instance = yolo
    elif (current_model == "ActDetYoloTiny"):
      module_instance = yolotiny
    elif (current_model == "FeatureExtractor"):
      module_instance = feature_extractor
    elif (current_model == "DeepSort"):
      module_instance = deepsort
    elif (current_model == "TubeManager"):
      module_instance = tube_manager
    elif (current_model == "ACAM"):
      module_instance = action_detector

    if (current_model == "DeepSort"):
      module_instance.PreProcess(request_input = request_input, istub = istub, tracker = my_tracker, my_lock = my_lock, grpc_flag = False)
    elif (current_model == "TubeManager"):
      module_instance.PreProcess(request_input = request_input, istub = istub, tube_manager = my_tube_manager, my_lock = my_lock, grpc_flag = False)
    else:
      module_instance.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
    module_instance.Apply()
    next_request = module_instance.PostProcess(grpc_flag = False)

    next_request['frame_info'] = request_input['frame_info']
    next_request['route_table'] = request_input['route_table']
    next_request['route_index'] = request_input['route_index'] + 1

    request_input = next_request

    # if (current_model == "DeepSort"):
      # print(request_input["deepsort_output"])
    # if (current_model == "TubeManager"):
      # print(request_input["temporal_rois_output"])
    if (current_model == "ACAM"):
      if (request_input["FINAL"] != "None@None"):
        print(request_input["FINAL"])

  end = time.time()
  duration = end - start
  print("Duration = %s" % duration)

  frame_id += 1
