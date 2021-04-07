# Caesar-ActDet

## How to Use
- Clone this repo to your machine ```git clone --recurse-submodules https://github.com/Rajrup/Caesar-Edge.git```
- Run ```python pipeline_new_reid.py``` (change TRACKER = "resnet" or "deepsort")
- Check if model loading works ```python load_model.py``` 

## Checkpoint Preparation
Find a zip of all the models: [Google Drive](https://drive.google.com/drive/folders/1eMfPOzYb2W-VUI2UikejhmZuX5aJ2aFW?usp=sharing)

All the NN model files should be put into the "checkpoints/" folder.

## Requirements
See ```requirements.txt``` for python packages.

## Components
One module's output will go to the next one
- Video Reader
- Object Detection ([SSD](https://github.com/balancap/SSD-Tensorflow), [YOLO](https://github.com/thtrieu/darkflow))
- Tracking ([DeepSORT](https://github.com/nwojke/deep_sort))
- Action Detection ([ACAM](https://github.com/oulutan/ACAM_Demo/blob/master/README.md))
- Triplet Reid ([REID](https://github.com/Rajrup/triplet-reid))


## Random
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
git clone https://github.com/tensorflow/models.git
git checkout r1.13.0
cd models/research
protoc object_detection/protos/*.proto --python_out=.