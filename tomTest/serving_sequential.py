import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
import threading

import time
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

object_detector = yolo

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

sess_id = "chain_actdet-000"
frame_id = 0
total_duration = 0.0
total_run = 10

special_duration = 0.0
special_count = 0

while (frame_id < total_run):

  start = time.time()
  frame_info = "%s-%s" % (sess_id, frame_id)

  # Read input
  frame_data = reader.PostProcess()
  if not frame_data:  # end of video 
    break 

  request_input = dict()
  request_input['client_input'] = frame_data['img']
  # request_input['frame_info'] = frame_info
  # request_input['route_table'] = route_table
  # request_input['route_index'] = route_index

  # Obj detection module
  object_detector.PreProcess(request_input = request_input, istub = istub, grpc_flag = False)
  object_detector.Apply()
  obj_det_data = object_detector.PostProcess(grpc_flag = False)

  # print(obj_det_data)
  # break

  # # Tracking module
  # feature_extractor.PreProcess(request_input = obj_det_data, istub = istub, grpc_flag = False)
  # feature_extractor.Apply()
  # feature_data = feature_extractor.PostProcess(grpc_flag = False)

  # # print(feature_data['reid_output'])
  # # break

  # tracker.PreProcess(request_input = feature_data, istub = istub, tracker = my_tracker, my_lock = my_lock, grpc_flag = False)
  # tracker.Apply()
  # track_data = tracker.PostProcess(grpc_flag = False)

  # # print(track_data['deepsort_output'])

  # # Action detection module 
  # track_data["frame_info"] = frame_info
  # tube_manager.PreProcess(request_input = track_data, istub = istub, tube_manager = my_tube_manager, my_lock = my_lock, grpc_flag = False)
  # tube_manager.Apply()
  # tube_data = tube_manager.PostProcess(grpc_flag = False)

  # # if ('meta' in tube_data):
  # #   print(tube_data['meta']['obj']['temporal_rois'])
  # # else:
  # #   print(tube_data)

  # special_start = time.time()

  # action_detector.PreProcess(request_input = tube_data, istub = istub, grpc_flag = False)
  # action_detector.Apply()
  # action_data = action_detector.PostProcess(grpc_flag = False)

  # special_end = time.time()

  # if (action_data['FINAL'] != "None@None"):
  #   # print(action_data['meta']['obj'])
  #   print(action_data['FINAL'])
  #   special_tmp = special_end - special_start
  #   special_duration += special_tmp
  #   special_count += 1
  #   print("ACAM takes %s" % str(special_tmp))
  #   print("ACAM end = %s" % str(special_end))

  frame_id += 1
  end = time.time()

  duration = end - start
  total_duration += duration
  print("This duration = %s" % str(duration))

print("Average duartion = %s" % str(total_duration / total_run))
# print("Average ACAM duration = %s" % str(special_duration / special_count))



