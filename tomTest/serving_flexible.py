import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

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

# yolo = YOLO()
# yolo.Setup()

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




# simple_route_table = "SSD-FeatureExtractor-DeepSort"
simple_route_table = "SSD-FeatureExtractor-DeepSort-TubeManager-ACAM"
route_table = simple_route_table

sess_id = "chain_actdet-000"
frame_id = -1

# while (frame_id < 32):
while (True):
  frame_id += 1
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
    elif (current_model == "FeatureExtractor"):
      module_instance = feature_extractor
    elif (current_model == "DeepSort"):
      module_instance = deepsort
    elif (current_model == "TubeManager"):
      module_instance = tube_manager
    elif (current_model == "ACAM"):
      module_instance = action_detector

    if (current_model == "DeepSort"):
      module_instance.PreProcess(request_input = request_input, istub = istub, tracker = my_tracker, grpc_flag = False)
    elif (current_model == "TubeManager"):
      module_instance.PreProcess(request_input = request_input, istub = istub, tube_manager = my_tube_manager, grpc_flag = False)
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




# frame_id = -1
# while (frame_id < 10):
#   frame_id += 1

#   # Read input
#   frame_data = reader.PostProcess()
#   if not frame_data:  # end of video 
#     break 

#   # Obj detection module
#   object_detector.PreProcess(request_input = frame_data, istub = istub, grpc_flag = False)
#   object_detector.Apply()
#   obj_det_data = object_detector.PostProcess(grpc_flag = False)

#   # print(obj_det_data)
#   # break

#   # Tracking module
#   feature_extractor.PreProcess(request_input = obj_det_data, istub = istub, grpc_flag = False)
#   feature_extractor.Apply()
#   feature_data = feature_extractor.PostProcess(grpc_flag = False)

#   # print(feature_data['reid_output'])
#   # break

#   tracker.PreProcess(request_input = feature_data, istub = istub, tracker = my_tracker, grpc_flag = False)
#   tracker.Apply()
#   track_data = tracker.PostProcess(grpc_flag = False)

#   # print(track_data['deepsort_output'])

#   # Action detection module 
#   tube_manager.PreProcess(track_data)
#   tube_manager.Apply()
#   tube_data = tube_manager.PostProcess()

#   if ('meta' in tube_data):
#     print(tube_data['meta']['obj']['temporal_rois'])
#   else:
#     print(tube_data)