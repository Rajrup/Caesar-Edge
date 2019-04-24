import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/')

# from modules_actdet.data_reader import DataReader
# from modules_actdet.object_detector_ssd import SSD
# from modules_actdet.object_detector_yolo import YOLO
# from modules_actdet.reid_extractor import FeatureExtractor
# from modules_actdet.tracker_deepsort import DeepSort
# from modules_actdet.tube_manager import TubeManager
# from modules_actdet.action_detector_acam import ACAM

if (sys.argv[1] == "original"):
  # original version
  from modules_actdet.data_reader import DataReader
  from modules_actdet.object_detector_ssd import SSD
  from modules_actdet.object_detector_yolo import YOLO
  from modules_actdet.reid_extractor import FeatureExtractor
  from modules_actdet.tracker_deepsort import DeepSort
  from modules_actdet.tube_manager import TubeManager
  from modules_actdet.action_detector_acam import ACAM

elif (sys.argv[1] == "serving"):
  # serving version
  from modules_actdet.data_reader import DataReader
  from modules_actdet.object_detector_ssd_serving import SSD
  from modules_actdet.object_detector_yolo_serving import YOLO
  from modules_actdet.reid_extractor_serving import FeatureExtractor
  from modules_actdet.tracker_deepsort_serving import DeepSort
  from modules_actdet.tube_manager_serving import TubeManager
  from modules_actdet.action_detector_acam_serving import ACAM

# ============ Video Input Modules ============
reader = DataReader()
reader.Setup("/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/indoor_two_ppl.avi")

# ============ Object Detection Modules ============
# ssd = SSD()
# ssd.Setup()

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

frame_id = -1
while(True):
# while (frame_id < 32):
    frame_id += 1

    # Read input
    frame_data = reader.PostProcess()
    if not frame_data:  # end of video 
        break 

    # Obj detection module
    object_detector.PreProcess(frame_data)
    object_detector.Apply()
    obj_det_data = object_detector.PostProcess()

    # print(obj_det_data)
    # break

    # Tracking module
    feature_extractor.PreProcess(obj_det_data)
    feature_extractor.Apply()
    feature_data = feature_extractor.PostProcess()

    # print(feature_data)
    # break

    tracker.PreProcess(feature_data)
    tracker.Apply()
    track_data = tracker.PostProcess()

    # print(track_data['meta'])

    # Action detection module 
    tube_manager.PreProcess(track_data)
    tube_manager.Apply()
    tube_data = tube_manager.PostProcess()

    # if ('meta' in tube_data):
    #   print(tube_data['meta']['obj']['temporal_rois'])
    # else:
    #   print(tube_data)

    action_detector.PreProcess(tube_data)
    action_detector.Apply()
    action_data = action_detector.PostProcess()

    if action_data:
        # print(action_data['meta']['obj'])
        print(action_data['meta'])
