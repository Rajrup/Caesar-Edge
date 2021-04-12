sudo docker run --gpus all -p 8500:8500 \
	--mount type=bind,source=/home/rajrup/Dropbox/Project/D2/Caesar-Edge/tf_servable,target=/models/tf_servable \
	-t --entrypoint=tensorflow_model_server tensorflow/serving:1.14.0-gpu --port=8500 \
	--platform_config_file=/models/tf_servable/platform.conf --model_config_file=/models/tf_servable/models.conf
