from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

class TubeManager:

  # initialize static variable here
  @staticmethod
  def Setup():
    TubeManager.cache_size = 32            # every result requires 32 frames
    TubeManager.action_freq = 16           # update the act result every 16 frames 

  def getTubeInput(self, raw_image, frame_id, deepsort_output):
    output = {'img': raw_image, 'meta': {'frame_id': frame_id}}
    output['meta']['obj'] = []
    if (len(deepsort_output) == 0):
      return output
    else:
      for tmp in deepsort_output.split('-'):
        tt = tmp.split('|')
        output['meta']['obj'].append({'box':[int(tt[0]), int(tt[1]), int(tt[2]), int(tt[3])], 'tid': int(tt[4])})
      return output

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    raw_image = tensor_util.MakeNdarray(request.inputs["raw_image"])
    deepsort_output = str(tensor_util.MakeNdarray(request.inputs["deepsort_output"]))
    frame_id = int(str(tensor_util.MakeNdarray(request.inputs["frame_info"])).split('-')[-1])

    tube_input = self.getTubeInput(raw_image, frame_id, deepsort_output)

    data_dict["tube_input"] = tube_input
    data_dict["frame_id"] = frame_id

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
      batched_data_dict["tube_input"] = []
      for data in data_array:
        batched_data_dict["tube_input"].append(data["tube_input"])

      batched_data_dict["frame_id"] = []
      for data in data_array:
        batched_data_dict["frame_id"].append(data["frame_id"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub, tube_manager, my_lock):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      # assume no batching...
      can_output = False

      my_lock.acquire()
      tube_manager.add_frame(batched_data_dict["tube_input"][0])

      if (batched_data_dict["frame_id"][0] % TubeManager.action_freq != 0 or batched_data_dict["frame_id"][0] < TubeManager.cache_size):
        pass
      elif (not tube_manager.has_new_tube()):
        pass
      else:
        frames, temporal_rois, norm_rois, actor_boxes = tube_manager.new_tube_data()
        can_output = True

      my_lock.release()

      if (can_output):
        batched_result_dict["frames"] = [frames]
        batched_result_dict["temporal_rois"] = [temporal_rois]
        batched_result_dict["norm_rois"] = [norm_rois]
        batched_result_dict["actor_boxes"] = [actor_boxes]
      else:
        batched_result_dict["frames"] = [None]
        batched_result_dict["temporal_rois"] = [None]
        batched_result_dict["norm_rois"] = [None]
        batched_result_dict["actor_boxes"] = [None]

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
        my_dict["frames"] = [batched_result_dict["frames"][i]]
        my_dict["temporal_rois"] = [batched_result_dict["temporal_rois"][i]]
        my_dict["norm_rois"] = [batched_result_dict["norm_rois"][i]]
        my_dict["actor_boxes"] = [batched_result_dict["actor_boxes"][i]]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []
    for i in range(len(result_dict[result_dict.keys()[0]])):
      if (result_dict["frames"][i] is not None):
        result_list.append({"frames": result_dict["frames"][i], "temporal_rois": result_dict["temporal_rois"][i], "norm_rois": result_dict["norm_rois"][i], "actor_boxes": result_dict["actor_boxes"][i]})
    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result):
    next_request = predict_pb2.PredictRequest()
    next_request.inputs["frames"].CopyFrom(
      tf.make_tensor_proto(result["frames"]))
    next_request.inputs["temporal_rois"].CopyFrom(
      tf.make_tensor_proto(result["temporal_rois"]))
    next_request.inputs["norm_rois"].CopyFrom(
      tf.make_tensor_proto(result["norm_rois"]))
    next_request.inputs["actor_boxes"].CopyFrom(
      tf.make_tensor_proto(result["actor_boxes"]))
    return next_request
