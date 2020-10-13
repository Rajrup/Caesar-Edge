import threading
import cv2
import grpc
import time
import numpy as np
import os
import pickle
import sys

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow_serving.apis import prediction_service_pb2_grpc

sys.path.append('/home/yitao/Documents/edge/D2-system/')
from utils_d2 import misc
from modules_d2.video_reader import VideoReader

sys.path.append(os.environ['CAESAR_EDGE_PATH'])
from modules_actdet.object_detector_ssd_inception_d2 import ActDetInception
from modules_actdet.reid_extractor_d2 import FeatureExtractor
from modules_actdet.deepsort_merged_d2 import DeepSort
from modules_actdet.tube_manager_d2 import TubeManager
from modules_actdet.action_detector_acam_d2 import ACAM

from deep_sort.tracker import Tracker
from deep_sort import nn_matching
# from modules_actdet.tube_manager_d2 import TubeManager
from modules_actdet.acam.manage_tube_d2 import TManager

ActDetInception.Setup()
FeatureExtractor.Setup()
DeepSort.Setup()
TubeManager.Setup()
ACAM.Setup()

ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

# module_name = "actdet_inception"
module_name = "actdet_reid"
# module_name = "actdet_deepsort"
# module_name = "actdet_tubemanager"
# module_name = "actdet_acam"

pickle_directory = "%s/pickle_d2/Caesar-Edge/%s" % (os.environ['RIM_DOCKER_SHARE'], module_name)
if not os.path.exists(pickle_directory):
  os.makedirs(pickle_directory)

batch_size = 4
parallel_level = 1
run_num = 1

def runBatch(batch_size, run_num, tid):
  start = time.time()

  reader = VideoReader()
  reader.Setup("%s/indoor_2min.mp4" % os.environ['CAESAR_EDGE_PATH'])

  if (module_name == "actdet_deepsort"):
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
    tracker = Tracker(metric, max_iou_distance = 0.7, max_age = 200, n_init = 4)
    my_lock = threading.Lock()
  elif (module_name == "actdet_tubemanager"):
    tube_manager = TManager(cache_size = TubeManager.cache_size, min_tube_len = TubeManager.action_freq)
    my_lock = threading.Lock()

  frame_id = 0
  batch_id = 0

  while (batch_id < run_num):
    module_instance = misc.prepareModuleInstance(module_name)
    data_array = []

    if (module_name == "actdet_inception"):
      for i in range(batch_size):
        client_input = reader.PostProcess()
        request = dict()
        request["client_input"] = client_input
        data_dict = module_instance.GetDataDict(request, grpc_flag = False)
        data_array.append(data_dict)
        frame_id += 1
    elif (module_name == "actdet_reid"):
      for i in range(batch_size):
        pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "Caesar-Edge", "actdet_inception"), str(frame_id + 1).zfill(3))
        request = pickle.load(open(pickle_input))
        data_dict = module_instance.GetDataDict(request, grpc_flag = False)
        data_array.append(data_dict)
        frame_id += 1
    elif (module_name == "actdet_deepsort"):
      pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "Caesar-Edge", "actdet_reid"), str(frame_id + 1).zfill(3))
      request = pickle.load(open(pickle_input))
      data_dict = module_instance.GetDataDict(request, grpc_flag = False)
      data_array.append(data_dict)
      frame_id += 1
    elif (module_name == "actdet_tubemanager"):
      pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "Caesar-Edge", "actdet_deepsort"), str(frame_id + 1).zfill(3))
      request = pickle.load(open(pickle_input))
      request["frame_info"] = "tmp-%d" % (frame_id + 1)
      data_dict = module_instance.GetDataDict(request, grpc_flag = False)
      data_array.append(data_dict)
      frame_id += 1
    elif (module_name == "actdet_acam"):
      pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "Caesar-Edge", "actdet_tubemanager"), str(32).zfill(3))
      request = pickle.load(open(pickle_input))
      data_dict = module_instance.GetDataDict(request, grpc_flag = False)
      data_array.append(data_dict)
      frame_id += 1

    batched_data_dict = module_instance.GetBatchedDataDict(data_array, batch_size)

    if (module_name == "actdet_deepsort"):
      batched_result_dict = module_instance.Apply(batched_data_dict, batch_size, istub, tracker, my_lock)
    elif (module_name == "actdet_tubemanager"):
      batched_result_dict = module_instance.Apply(batched_data_dict, batch_size, istub, tube_manager, my_lock)
    else:
      batched_result_dict = module_instance.Apply(batched_data_dict, batch_size, istub)

    batched_result_array = module_instance.GetBatchedResultArray(batched_result_dict, batch_size)

    for i in range(len(batched_result_array)):
      # deal with the outputs of the ith input in the batch
      result_dict = batched_result_array[i]

      # each input might have more than one outputs
      result_list = module_instance.GetResultList(result_dict)

      for result in result_list:
        next_request = module_instance.GetNextRequest(result, grpc_flag = False)

        if (module_name == "actdet_inception"):
          print(next_request["objdet_output"])
        #   pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
        #   with open(pickle_output, 'w') as f:
        #     pickle.dump(next_request, f)

        if (module_name == "actdet_reid"):
          print(next_request["features"])
        #   pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
        #   with open(pickle_output, 'w') as f:
        #     pickle.dump(next_request, f)

        # if (module_name == "actdet_deepsort"):
        #   print(next_request["deepsort_output"])
        #   pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
        #   with open(pickle_output, 'w') as f:
        #     pickle.dump(next_request, f)

        # if (module_name == "actdet_tubemanager"):
        #   print(next_request["actor_boxes"])
        #   pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
        #   with open(pickle_output, 'w') as f:
        #     pickle.dump(next_request, f)

        # if (module_name == "actdet_acam"):
        #   print(next_request["actdet_output"])

    batch_id += 1

  end = time.time()
  print("[Thread-%d] it takes %.3f sec to run %d batches of batch size %d" % (tid, end - start, run_num, batch_size))


# ========================================================================================================================

start = time.time()

thread_pool = []
for i in range(parallel_level):
  t = threading.Thread(target = runBatch, args = (batch_size, run_num, i))
  thread_pool.append(t)
  t.start()

for t in thread_pool:
  t.join()

end = time.time()
print("overall time = %.3f sec" % (end - start))
