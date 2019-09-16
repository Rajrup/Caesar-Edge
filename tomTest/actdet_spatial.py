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

input_fps = int(sys.argv[1])
total_frame = 120

def runFrame(measure_module, request_input, frame_id):
  # if (measure_module == "SSD"):
  #   module_instance = ssd
  if (measure_module == "Yolo"):
    module_instance = yolo
  elif (measure_module == "YoloTiny"):
    module_instance = yolotiny
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

  print("Finished frame %d for module %s" % (frame_id, measure_module))

  if (frame_id == 10):
    global stime
    stime = time.time()
  elif (frame_id == total_frame - 1):
    global etime
    etime = time.time()


if True:
  # get input
  if (measure_module == "SSD" or measure_module == "Yolo" or measure_module == "YoloTiny"):
    frame_info = "%s-%s" % (sess_id, "32")
    route_index = 0
    frame_data = reader.PostProcess()
    request_input = dict()
    request_input['client_input'] = frame_data['img']
    request_input['frame_info'] = frame_info
    request_input['route_table'] = route_table
    request_input['route_index'] = route_index
  elif (measure_module == "FeatureExtractor"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(32).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "DeepSort"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(32).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "TubeManager"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(32).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)
  elif (measure_module == "ACAM"):
    pickle_input = "%s/%s" % ("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/pickle_tmp/%s" % findPreviousModule(route_table, measure_module), str(32).zfill(3))
    with open(pickle_input) as f:
      request_input = pickle.load(f)

while frame_id < total_frame:
  # frame_thread = threading.Thread(target = runFrame, args = (measure_module, sess_id, frame_id, reader,))
  frame_thread = threading.Thread(target = runFrame, args = (measure_module, request_input, frame_id,))
  frame_thread.start()

  time.sleep(1.0/input_fps)
  frame_id += 1

try:
  while True:
    time.sleep(60 * 60 * 24)
except KeyboardInterrupt:
  print("\nEnd by keyboard interrupt")
  print("<%f, %f> = %f over %d frames with fps of %f" % (float(stime), float(etime), float(etime) - float(stime), total_frame, (total_frame - 1 - 10) / (float(etime) - float(stime))))
