import sys
import os
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

sys.path.append(os.environ['D2_SYSTEM_PATH'])
from modules_d2.video_reader import VideoReader

sys.path.append(os.environ['CAESAR_EDGE_PATH'])
from modules_actdet.object_detector_yolo_d2 import YOLO

actdet_reader = VideoReader()
actdet_reader.Setup("%s/indoor_2min.mp4" % os.environ['CAESAR_EDGE_PATH'])
source_reader = actdet_reader

yolo = YOLO()
yolo.Setup()
object_detector = yolo

ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

frame_id = 0

while (frame_id < 160):

  client_input = source_reader.PostProcess()
  # print(client_input.shape)

  request_dict = dict()
  request_dict["client_input"] = client_input

  object_detector.PreProcess(request_dict = request_dict, istub = istub)
  object_detector.Apply()
  next_request_list = object_detector.PostProcess()

  for next_request in next_request_list:
    print(next_request["objdet_output"])

  frame_id += 1