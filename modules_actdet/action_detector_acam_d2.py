from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

import modules_actdet.acam.action_detector_serving as act

import numpy as np 

PRINT_TOP_K = 5      # show the top 5 possible action for current second 

class ACAM:

  # initialize static variable here
  @staticmethod
  def Setup():
    pass

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request, grpc_flag):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    if (grpc_flag):
      frames = tensor_util.MakeNdarray(request.inputs["frames"])
      temporal_rois = tensor_util.MakeNdarray(request.inputs["temporal_rois"])
      norm_rois = tensor_util.MakeNdarray(request.inputs["norm_rois"])
      actor_boxes = tensor_util.MakeNdarray(request.inputs["actor_boxes"])
    else:
      frames = request["frames"]
      temporal_rois = request["temporal_rois"]
      norm_rois = request["norm_rois"]
      actor_boxes = request["actor_boxes"]

    data_dict["frames"] = frames
    data_dict["temporal_rois"] = temporal_rois
    data_dict["norm_rois"] = norm_rois
    data_dict["actor_boxes"] = actor_boxes

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
      batched_data_dict["frames"] = []
      for data in data_array:
        batched_data_dict["frames"].append(data["frames"])

      batched_data_dict["temporal_rois"] = []
      for data in data_array:
        batched_data_dict["temporal_rois"].append(data["temporal_rois"])

      batched_data_dict["norm_rois"] = []
      for data in data_array:
        batched_data_dict["norm_rois"].append(data["norm_rois"])

      batched_data_dict["actor_boxes"] = []
      for data in data_array:
        batched_data_dict["actor_boxes"].append(data["actor_boxes"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      # assuming no batching
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'actdet_acam'
      request.model_spec.signature_name = 'predict_images'

      tube_num = len(batched_data_dict["actor_boxes"][0])
      nptmp1 = np.zeros(tube_num)
      nptmp2 = np.arange(tube_num)

      # print(batched_data_dict["frames"][0])

      request.inputs['updated_frames'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batched_data_dict["frames"][0], dtype = tf.float32, shape = batched_data_dict["frames"][0].shape))
      request.inputs['temporal_rois'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batched_data_dict["temporal_rois"][0], dtype = tf.float32, shape = batched_data_dict["temporal_rois"][0].shape))
      request.inputs['temporal_roi_batch_indices'].CopyFrom(
        tf.contrib.util.make_tensor_proto(nptmp1, dtype = tf.int32, shape = nptmp1.shape))
      request.inputs['rois'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batched_data_dict["norm_rois"][0], dtype = tf.float32, shape = batched_data_dict["norm_rois"][0].shape))
      request.inputs['roi_batch_indices'].CopyFrom(
        tf.contrib.util.make_tensor_proto(nptmp2, dtype = tf.int32, shape = nptmp2.shape))

      result = istub.Predict(request, 10.0)
      probs = tensor_util.MakeNdarray(result.outputs['output'])

      if (not len(probs)):
        abstr = "None"
        resstr = "None"
      else:
        abstr = ""
        for ab in batched_data_dict["actor_boxes"][0]:
          abstr += "%s-" % ab
        abstr = abstr[:-1]
        resstr = ""
        for i in range(tube_num):
          act_probs = probs[i]
          order = np.argsort(act_probs)[::-1]
          for pp in range(PRINT_TOP_K):
            resstr += "%s|%s|" % (str(act.ACTION_STRINGS[order[pp]]), str(act_probs[order[pp]]))
          resstr = resstr[:-1]
          resstr += '-'
        resstr = resstr[:-1]

      batched_result_dict["actdet_output"] = ["%s@%s" % (abstr, resstr)]

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
        my_dict["actdet_output"] = [batched_result_dict["actdet_output"][i]]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []
    for i in range(len(result_dict[result_dict.keys()[0]])):
      result_list.append({"actdet_output": result_dict["actdet_output"][i]})

    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["actdet_output"].CopyFrom(
        tf.make_tensor_proto(result["actdet_output"]))
    else:
      next_request = dict()
      next_request["actdet_output"] = result["actdet_output"]
    return next_request
