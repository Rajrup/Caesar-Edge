# Darkflow should be installed from: https://github.com/sugartom/darkflow
from darkflow.net.build import TFNet

YOLO_PEOPLE_LABEL = 'person'
YOLO_THRES = 0.4

YOLO_CONFIG = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/cfg'
YOLO_MODEL = '/home/yitao/Documents/fun-project/tensorflow-related/Caesar-Edge/cfg/yolo.cfg'
YOLO_WEIGHTS = '/home/yitao/Downloads/tmp/docker-share/module_actdet/checkpoints/yolo/yolo.weights'

class YOLO:

  @staticmethod
  def Setup():
    opt = { "config": YOLO_CONFIG,  
            "model": YOLO_MODEL, 
            "load": YOLO_WEIGHTS, 
            "threshold": YOLO_THRES
          }
    YOLO.tfnet = TFNet(opt)

  def PreProcess(self, request_dict, istub):
    self.client_input = request_dict["client_input"]
    self.istub = istub

  def Apply(self):
    self.dets = YOLO.tfnet.return_predict(self.client_input, "actdet_yolo", self.istub)

    output = ""
    for d in self.dets:
      if d['label'] != YOLO_PEOPLE_LABEL:
        continue
      output += "%s|%s|%s|%s|%s|%s-" % (str(d['topleft']['x']), str(d['topleft']['y']), str(d['bottomright']['x']), str(d['bottomright']['y']), str(d['confidence']), str(d['label']))

    self.output = output[:-1]

  def PostProcess(self):
    next_request_dict = dict()
    next_request_dict["client_input"] = self.client_input
    next_request_dict["objdet_output"] = self.output

    return [next_request_dict]
