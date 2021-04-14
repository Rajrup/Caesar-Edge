# Caesar-ActDet

## How to Use

- Clone this repo to your machine ```git clone --recurse-submodules https://github.com/Rajrup/Caesar-Edge.git```
- Install the python dependencies (see below)
- Download the checkpoints and tensorflow servables (see below)
- Double check the model paths
- Ready to run ```pipeline_new_reid.py```

## Checkpoint Preparation

- Find a zip of all the models: [Google Drive](https://drive.google.com/drive/folders/1eMfPOzYb2W-VUI2UikejhmZuX5aJ2aFW?usp=sharing)
- Extract the Tensorflow model files into ```checkpoints/``` folder.
  - Deepsort: Path will look this - ```./checkpoints/deepsort```
  - SSD: Path will look this - ```./checkpoints/ssd_mobilenet_v1_coco_2017_11_17```
  - Triplet-Reid: Path will look this - ```./checkpoints/triplet-reid```

- Extract the Tensorflow Serving model files into ```tf_servable/``` folder.
- Change the source in ```run_tf_server.sh``` to absolute path to ```tf_servable/``` folder.

## Requirements
This code has been tested in ```Python 3.7```.
See ```requirements.txt``` for python packages.

```bash
pip install -r requirements.txt
```

## Running Pipeline

- Tensorflow Pipeline:
  - Run with Deepsort Reid feature extractor

    ```python
    python pipeline_new_reid.py original deepsort_reid
    ```

  - Run with Triplet Reid feature extractor

    ```python
    python pipeline_new_reid.py original resnet_reid
    ```

- Tensorflow Serving Pipeline:
  - Run serving in docker

     ```bash
     chmod +x run_tf_server.sh
    ./run_tf_server.sh
    ```

  - Run client with Deepsort Reid:

    ```python
    python pipeline_new_reid.py serving deepsort_reid
    ```

  - Run client with Triplet Reid:

    ```python
    python pipeline_new_reid.py serving resnet_reid
    ```

## Components

One module's output will go to the next one

- Video Reader
- Object Detection ([SSD](https://github.com/balancap/SSD-Tensorflow), [YOLO](https://github.com/thtrieu/darkflow))
- Tracking ([DeepSORT](https://github.com/nwojke/deep_sort))
- Action Detection ([ACAM](https://github.com/oulutan/ACAM_Demo/blob/master/README.md))
- Triplet Reid ([REID](https://github.com/Rajrup/triplet-reid))