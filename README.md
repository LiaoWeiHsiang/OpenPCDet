# Improve LiDAR Object Detection Using Point Cloud Map

<!-- ![](detection_long_range_demo.gif) -->
<video src="clip.mp4" controls></video>
# Abstract
We propose a novel method that combine Point Cloud Map with LiDAR object detection method and extract the environment information using Graph Neural Network to improve the detection performance on long-range objects and reduce false positives. We implement our method base on a excellent object detector PV-RCNN and test on NuScenes dataset.
# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 16.04)
* Python 3.6
* PyTorch 1.8
* CUDA 10.0
* [`spconv v1.2.1`](https://github.com/traveller59/spconv/tree/v1.2.1)

### Install
a. Clone this repository.
```shell
git clone https://github.com/open-mmlab/OpenPCDet.git
```

b. Install the dependent libraries as follows:
* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). We provide three ways to install it. Choose a method to install it.
    1.    We use [`spconv v1.2.1`](https://github.com/traveller59/spconv/tree/v1.2.1) to impelment  our work.  Please note the version and the branch! Follow the repo to install it.
    2.    Please download package from (https://drive.google.com/file/d/1HbLTW5_cyM7QwWr9ycIEl-xxCN2Y7bJn/view?usp=sharing) and unzip it. Install the package following the instruct of [`spconv v1.2.1`](https://github.com/traveller59/spconv/tree/v1.2.1).
    3.    If use Python3.6, pip install the package downloaded from https://drive.google.com/file/d/1va403FxPuVuvVXCAJXCYW3dpRXFfaS8b/view?usp=sharing and pip install it. 

  
c. Install this project's library and its dependent libraries by running the following command:
```shell
python setup.py develop
```

# Getting Started

## Dataset Preparation
### NuScenes Dataset
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and 
organize the downloaded files as follows: 
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```

* Generate the data infos by running the following command (it may take several hours): 
```python 
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
```
* Download the NuScenes maps: https://drive.google.com/file/d/1LgbuK1PsE4Pakg4kabGBdle6yDXqFv1_/view?usp=sharing
```
OpenPCDet
├── maps
│   ├──cfg
│   │   ├──map_by_scenes_v7_Downsampling0.1_no_ground_ieflat
│   │   │   ├──boston-seaport      
│   │   │   ├──singapore-hollandvillage
│   │   │   ├──singapore-onenorth
│   │   │   ├──singapore-queenstown
├── data
├── pcdet
├── tools
```


## Pretrained Models

The pretrained models can be obtained on: https://drive.google.com/file/d/1TopnnbRCMH5_3G0hyKuTKeP8Y55OSoD3/view?usp=sharing

```
OpenPCDet
├── output
│   ├──cfg
│   │   ├──nuscenes_models
│   │   │   ├──...
├── data
├── pcdet
├── tools
```
## Training & Testing
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ./cfgs/nuscenes_models/pv_rcnn.yaml --ckpt ../output/cfgs/nuscenes_models/pv_rcnn/default/ckpt/checkpoint_epoch_33.pth --batch_size 1 --map_shift 0
```
* Train a model

```
python train.py --cfg_file ./cfgs/nuscenes_models/pv_rcnn.yaml --map_shift 0
```
