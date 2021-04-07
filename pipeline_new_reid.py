from modules_actdet.data_reader import DataReader
# from modules_actdet.object_detector_ssd import SSD
from modules_actdet.object_detector_ssd_mobilenet import SSD
# from modules_actdet.object_detector_yolo import YOLO
from modules_actdet.reid_extractor import FeatureExtractor
from modules_actdet.reid_extractor_resnet import FeatureExtractor2
from modules_actdet.tracker_deepsort import DeepSort
# from modules_actdet.tube_manager import TubeManager
# from modules_actdet.action_detector_acam import ACAM
import sys
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time

TRACKER = "resnet"

# ============ Video Input Modules ============
reader = DataReader()
reader.Setup("./video/indoor_two_ppl.avi")

# ============ Object Detection Modules ============
ssd = SSD()
ssd.Setup()

# yolo = YOLO()
# yolo.Setup()

object_detector = ssd

# ============ Tracking Modules ============
if TRACKER == "deepsort":
    feature_extractor = FeatureExtractor()
else:
    feature_extractor = FeatureExtractor2()
feature_extractor.Setup()

deepsort = DeepSort()
deepsort.Setup()

tracker = deepsort

track_output = "./video/tracker_{}.avi".format(TRACKER)

# # ============ Action Detection Modules ============
# tube_manager = TubeManager()
# tube_manager.Setup()

# acam = ACAM()
# acam.Setup()

# action_detector = acam

width = int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(reader.cap.get(cv2.CAP_PROP_FPS))
fourcc = int(reader.cap.get(cv2.CAP_PROP_FOURCC))
track_out = cv2.VideoWriter(track_output, fourcc, fps, (width, height))

#initialize color map
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
fps = 0.0

try:

    while(True):

        t1 = time.time()
        # Read input
        frame_data = reader.PostProcess()
        if not frame_data:  # end of video 
            break 

        # Obj detection module
        object_detector.PreProcess(frame_data)
        object_detector.Apply()
        obj_det_data = object_detector.PostProcess()

        # Tracking module
        feature_extractor.PreProcess(obj_det_data)
        feature_extractor.Apply()
        feature_data = feature_extractor.PostProcess()

        for feat in feature_data['meta']['obj']:
            print(feat['feature'].shape)

        tracker.PreProcess(feature_data)
        tracker.Apply()
        track_data = tracker.PostProcess()

        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        img = track_data['img']
        frame_id = track_data['meta']['frame_id']

        for track in track_data['meta']['obj']:
            left, top, width, height = track['box']
            bbox = [top, left, top + height, left + width]
            track_id = track['tid']
            color = colors[int(track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len("Track ID: ") + len(str(track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, "Track ID: " + str(track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255), 2)

        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if track_out:
            track_out.write(img)

        # print(type(track_data))

except KeyboardInterrupt:
    if track_out:
        track_out.release()

    
    #     # Action detection module 
    #     tube_manager.PreProcess(track_data)
    #     tube_manager.Apply()
    #     tube_data = tube_manager.PostProcess()

    #     action_detector.PreProcess(tube_data)
    #     action_detector.Apply()
    #     action_data = action_detector.PostProcess()

    #     if action_data:
    #         # print(action_data['meta']['obj'])
    #         print(action_data['meta'])
