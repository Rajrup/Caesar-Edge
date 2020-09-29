






class FeatureExtractor:

  @staticmethod
  def Setup():
    pass

  def PreProcess(self, request_dict, istub):
    self.istub = istub
    self.encoder = create_box_encoder(self.istub, batch_size=16)

    self.client_input = request_dict["client_input"]
    self.objdet_output = request_dict["objdet_output"]

    self.ds_boxes = []
    self.scores = []

    for b in self.objdet_output.split('-'):
      tmp = b.split('|')
      b0 = int(tmp[0])
      b1 = int(tmp[1])
      b2 = int(tmp[2])
      b3 = int(tmp[3])
      b4 = float(tmp[4])
      self.ds_boxes.append([b0, b1, b2 - b0, b3 - b1])
      self.scores.append(b4)

    def Apply(self):
      self.features = self.encoder(self.client_input, self.ds_boxes)

    def PostProcess(self):
      next_request_dict = dict()
      next_request_dict["client_input"] = self.client_input
      