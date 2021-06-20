# Caesar-Tracking

## How to Use

- Clone this repo to your machine ```git clone --recurse-submodules https://github.com/Rajrup/Caesar-Edge.git```
- Install the python dependencies (see below)
- Download the checkpoints and tensorflow servables (see below)
- Double check the model paths
- Ready to run ```pipeline_new_reid.py```

## Checkpoint Preparation

- Find a zip of all the models: [Google Drive](https://drive.google.com/drive/folders/1eMfPOzYb2W-VUI2UikejhmZuX5aJ2aFW?usp=sharing)
- Extract the Tensorflow model files into ```checkpoints/``` folder under the root directory.
  - Deepsort: Path will look this - ```./checkpoints/deepsort```
  - SSD: Path will look this - ```./checkpoints/ssd_mobilenet_v1_coco_2017_11_17```
  - Triplet-Reid: Path will look this - ```./checkpoints/triplet-reid```

- Extract the Tensorflow Serving model files into ```tf_servable/``` folder under the root directory.
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
  - Find the output fo the trackers in ```video``` folder.

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
- Object Detection ([SSD](https://github.com/balancap/SSD-Tensorflow)), We can also replace SSD with [YOLO](https://github.com/thtrieu/darkflow)
- Tracking 
  - Deepsort Reid ([DeepSORT](https://github.com/nwojke/deep_sort))
  - Triplet Reid ([Triplet-Reid](https://github.com/Rajrup/triplet-reid))