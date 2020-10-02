from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

import pickle

class DeepSortSequencer:

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
    detection_list = pickle.loads(str(tensor_util.MakeNdarray(request.inputs["detection_list"])))

    # for detection in detection_list:
      # print("tlwh = %s, confidence = %s" % (detection.tlwh, detection.confidence))

    data_dict["raw_image"] = raw_image
    data_dict["detection_list"] = detection_list

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
      batched_data_dict["raw_image"] = []
      for data in data_array:
        batched_data_dict["raw_image"].append(data["raw_image"])

      batched_data_dict["detection_list"] = []
      for data in data_array:
        batched_data_dict["detection_list"].append(data["detection_list"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub, tracker, my_lock):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      my_lock.acquire()

      tracker.predict()
      tracker.update(batched_data_dict["detection_list"][0])

      # print(len(tracker.tracks))

      output = ""
      for tk in tracker.tracks:
        # print("tk.is_confirmed() = %s, tk.time_since_update = %s" % (tk.is_confirmed(), tk.time_since_update))
        if not tk.is_confirmed() or tk.time_since_update > 1:
          continue
        left, top, width, height = map(int, tk.to_tlwh())
        track_id = tk.track_id
        output += "%s|%s|%s|%s|%s-" % (str(left), str(top), str(width), str(height), str(track_id))

      output = output[:-1]

      my_lock.release()

      batched_result_dict["deepsort_output"] = [output]
      batched_result_dict["raw_image"] = batched_data_dict["raw_image"]

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
        my_dict["raw_image"] = [batched_result_dict["raw_image"][i]]
        my_dict["deepsort_output"] = [batched_result_dict["deepsort_output"][i]]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []
    for i in range(len(result_dict[result_dict.keys()[0]])):
      result_list.append({"deepsort_output": result_dict["deepsort_output"][i], "raw_image": result_dict["deepsort_output"][i]})
    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result):
    # print(result["deepsort_output"])
    next_request = predict_pb2.PredictRequest()
    next_request.inputs['raw_image'].CopyFrom(
      tf.make_tensor_proto(result["raw_image"]))
    next_request.inputs["deepsort_output"].CopyFrom(
      tf.make_tensor_proto(result["deepsort_output"]))
    return next_request
