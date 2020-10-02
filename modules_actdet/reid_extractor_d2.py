from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

import sys
import os
sys.path.append("%s/modules_actdet/deep_sort" % os.environ['CAESAR_EDGE_PATH'])
from tools.generate_detections_serving import create_box_encoder

# import pickle

class FeatureExtractor:

  # initialize static variable here
  @staticmethod
  def Setup():
    pass

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    raw_image = tensor_util.MakeNdarray(request.inputs["raw_image"]) 
    objdet_output = str(tensor_util.MakeNdarray(request.inputs["objdet_output"]))
    # print("[Debug] objdet_output = %s" % objdet_output)

    ds_boxes = []
    scores = []

    for b in objdet_output.split('-'):
      tmp = b.split('|')
      b0 = int(tmp[0])
      b1 = int(tmp[1])
      b2 = int(tmp[2])
      b3 = int(tmp[3])
      b4 = float(tmp[4])
      ds_boxes.append([b0, b1, b2 - b0, b3 - b1])
      scores.append(b4)

    data_dict["ds_boxes"] = ds_boxes
    data_dict["scores"] = scores
    data_dict["raw_image"] = raw_image
    data_dict["objdet_output"] = objdet_output

    return data_dict

  # for an array of requests from a batch, convert them to a dict,
  # where each key has a lit of values
  # input: data_array = [{"image": image1, "meta": meta1}, {"image": image2, "meta": meta2}]
  # output: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  def GetBatchedDataDict(self, data_array, batch_size):
    if (len(data_array) != batch_size):
      print("[Error] GetBatchedDataDict() batch size not matched...")
      return None
    else:
      batched_data_dict = dict()

      # for each key in data_array[0], convert it to batched_data_dict[key][]
      batched_data_dict["ds_boxes"] = []
      for data in data_array:
        batched_data_dict["ds_boxes"].append(data["ds_boxes"])

      batched_data_dict["scores"] = []
      for data in data_array:
        batched_data_dict["scores"].append(data["scores"])

      batched_data_dict["raw_image"] = []
      for data in data_array:
        batched_data_dict["raw_image"].append(data["raw_image"])

      batched_data_dict["objdet_output"] = []
      for data in data_array:
        batched_data_dict["objdet_output"].append(data["objdet_output"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      # assume no batching for reid_extractor (GPU)
      # To-do: should we enable batching?
      encoder = create_box_encoder(istub, batch_size = 16)
      features = encoder(batched_data_dict["raw_image"][0], batched_data_dict["ds_boxes"][0])

      batched_result_dict["features"] = [features]
      batched_result_dict["raw_image"] = batched_data_dict["raw_image"]
      batched_result_dict["objdet_output"] = batched_data_dict["objdet_output"]

      return batched_result_dict

  # input: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  # output: batched_result_array = [{"bounding_boxes": [bb1_in_image1, bb2_in_image1]}, {"bounding_boxes": [bb1_in_image2]}]
  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict[batched_result_dict.keys()[0]])):
      print("[Error] GetBatchedResultArray() batch size not matched...")
      return None
    else:
      batched_result_array = []

      for i in range(batch_size):
        my_dict = dict()
        my_dict["features"] = [batched_result_dict["features"][i]]
        my_dict["raw_image"] = [batched_result_dict["raw_image"][i]]
        my_dict["objdet_output"] = [batched_result_dict["objdet_output"][i]]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []
    for i in range(len(result_dict[result_dict.keys()[0]])):
      result_list.append({"features": result_dict["features"][i], "raw_image": result_dict["raw_image"][i], "objdet_output": result_dict["objdet_output"][i]})
    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result):
    # print(result["features"])

    next_request = predict_pb2.PredictRequest()
    next_request.inputs['raw_image'].CopyFrom(
      tf.make_tensor_proto(result["raw_image"]))
    next_request.inputs["objdet_output"].CopyFrom(
      tf.make_tensor_proto(result["objdet_output"]))
    # next_request.inputs["features"].CopyFrom(
    #   tf.make_tensor_proto(pickle.dumps(result["features"])))
    next_request.inputs["features"].CopyFrom(
      tf.make_tensor_proto(result["features"]))
    return next_request
