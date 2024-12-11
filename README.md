# Learning Bird’s Eye View Scene Graph and Knowledge-Inspired Policy for Embodied Visual Navigation


We propose BevNav framework to solve these issues by three parts: (i) we introduce a novel Bird’s Eye View (BEV) scene graph (BevSG) that utilizes multi-view 2D information transformed into 3D under the supervision of 3D detection to encode scene layouts and geometric clues. It can distinguish multi-view
semantically similar objects and make plans in this graph. (ii) we propose BEV-BLIP contrastive learning that aligns the BEV and language grounding inputs transferring constrain commonsense knowledge in pre-trained models without other training in the environments. (iii) we design BEV-based view search navigation policy, which encourages representations that encode the semantics, relationships, and positional information of objects. 




## Setup
- *Dependeces*: We use earlier (`0.2.2`) versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim/tree/v0.2.2) and [habitat-lab](https://github.com/facebookresearch/habitat-lab/tree/v0.2.2). Other related depencese can be found in `requirements.txt`. 

- *Data (MatterPort3D)*: Please download the scene dataset and the episode dataset from [habitat-lab/DATASETS.md](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset). Then organize the files as follows:
```
3dNav/
  data/
    scene_datasets/
        mp3d/
    episode_datasets/
        objectnav_mp3d_v1/
```

## Installation
The implementation of BEV Detection is built on [MMDetection3D v0.17.1](https://github.com/open-mmlab/mmdetection3d). Please follow [BEVFormer](https://github.com/fundamentalvision/BEVFormer) for installation. 

The implementation of VN is built on the latest version of [Matterport3D simulators](https://github.com/peteanderson80/Matterport3DSimulator):
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

Many thanks to the contributors for their great efforts.

## Dataset Preparation
The dataset is based on indoor RGB images from [Matterport3D](https://niessner.github.io/Matterport/). Please fill and sign the [Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) agreement form and send it to matterport3d@googlegroups.com to request access to the dataset. 

Note that we use the undistorted_color_images for BEV Detection. Camera parameters (word-to-pixel matrix) are from undistorted_camera_parameters. The 3D box annotations can be available in mp3dbev/data. For VLN, please follow [VLN-DUET](https://github.com/cshizhe/VLN-DUET) for more details, including processed annotations, features and pretrained models of REVERIE, R2R and R4R datasets.



## Extracting Features
Please follow the [scripts](https://github.com/cshizhe/VLN-HAMT/tree/main/preprocess) to extract visual features for both undistorted_color_images (for BEV Detection) and matterport_skybox_images (for VLN, optional). Note that all the ViT features of undistorted_color_images should be used (not only the [CLS] token, about 130 GB). Please note this line since different version of [timm](https://github.com/huggingface/pytorch-image-models) models have different output: 
```
b_fts = model.forward_features(images[k: k+args.batch_size])
```

## BEV Detection
```shell
cd mp3dbev/
# multi-gpu train
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=${PORT:id} ./tools/dist_train.sh ./projects/configs/bevformer/mp3dbev.py 4

# multi-gpu test
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=${PORT:id} ./tools/dist_test.sh ./projects/configs/bevformer/mp3dbev.py ./path/to/ckpts.pth 4

# inference for BEV features
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=${PORT:id} ./tools/dist_test.sh ./projects/configs/bevformer/getbev.py ./path/to/ckpts.pth 4
```
Please also see train and inference for the detailed [usage](https://github.com/open-mmlab/mmdetection3d) of MMDetection3D.


## Getting Started
For environment setup and dataset preparation, please follow:
* [Installation](./docs/installation.md)

For evaluation, please follow:
* [Evaluation](./docs/run.md)

## Training and Evaluating:

We provide scripts for quick training and evaluation. The parameters can be found in [sh_train_mp3d.sh](sh_train_mp3d.sh) and [sh_eval.sh](sh_eval.sh), You can modify these parameters to customize them according to your specific requirements.
```
sh sh_train_mp3d.sh # training 
sh sh_eval.sh # evaluating
```


