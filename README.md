# BEVFusion

### [website](http://bevfusion.mit.edu/) | [paper](https://arxiv.org/abs/2205.13542) | [video](https://www.youtube.com/watch?v=uCAka90si9E)

![demo](assets/demo.gif)

## News

- **(2024/5)** BEVFusion is integrated into NVIDIA [DeepStream](https://developer.nvidia.com/blog/nvidia-deepstream-7-0-milestone-release-for-next-gen-vision-ai-development/) for sensor fusion.
- **(2023/5)** NVIDIA provides a [TensorRT deployment solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion) of BEVFusion, achieving 25 FPS on Jetson Orin.
- **(2023/4)** BEVFusion ranks first on [Argoverse](https://eval.ai/web/challenges/challenge-page/1710/overview) 3D object detection leaderboard among all solutions.
- **(2023/1)** BEVFusion is integrated into [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion).
- **(2023/1)** BEVFusion is accepted to ICRA 2023!
- **(2022/8)** BEVFusion ranks first on [Waymo](https://waymo.com/open/challenges/2020/3d-detection/) 3D object detection leaderboard among all solutions.
- **(2022/6)** BEVFusion ranks first on [nuScenes](https://nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any) 3D object detection leaderboard among all solutions.
- **(2022/6)** BEVFusion ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) 3D object detection leaderboard among all solutions.

## Abstract

Multi-sensor fusion is essential for an accurate and reliable autonomous driving system. Recent approaches are based on point-level fusion: augmenting the LiDAR point cloud with camera features. However, the camera-to-LiDAR projection throws away the semantic density of camera features, hindering the effectiveness of such methods, especially for semantic-oriented tasks (such as 3D scene segmentation). In this paper, we break this deeply-rooted convention with BEVFusion, an efficient and generic multi-task multi-sensor fusion framework. It unifies multi-modal features in the shared bird's-eye view (BEV) representation space, which nicely preserves both geometric and semantic information. To achieve this, we diagnose and lift key efficiency bottlenecks in the view transformation with optimized BEV pooling, reducing latency by more than **40x**. BEVFusion is fundamentally task-agnostic and seamlessly supports different 3D perception tasks with almost no architectural changes. It establishes the new state of the art on the nuScenes benchmark, achieving **1.3%** higher mAP and NDS on 3D object detection and **13.6%** higher mIoU on BEV map segmentation, with **1.9x** lower computation cost.

## Results

### 3D Object Detection (on Waymo test)

|   Model   | mAP-L1 | mAPH-L1  | mAP-L2  | mAPH-L2  |
| :-------: | :------: | :--: | :--: | :--: |
| [BEVFusion](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&challengeId=DETECTION_3D&emailId=f58eed96-8bb3&timestamp=1658347965704580) |    82.72   |  81.35  | 77.65  |  76.33 |
| [BEVFusion-TTA](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&challengeId=DETECTION_3D&emailId=94ddc185-d2ce&timestamp=1663562767759105) | 86.04    |  84.76 | 81.22  |  79.97 |

Here, BEVFusion only uses a single model without any test time augmentation. BEVFusion-TTA uses single model with test-time augmentation and no model ensembling is applied. 

### 3D Object Detection (on nuScenes test)

|   Model   | Modality | mAP  | NDS  |
| :-------: | :------: | :--: | :--: |
| BEVFusion-e |   C+L    | 74.99 | 76.09 |
| BEVFusion |   C+L    | 70.23 | 72.88 |
| BEVFusion-base* |   C+L    | 71.72 | 73.83 |

*: We scaled up MACs of the model to match the computation cost of concurrent work.

### 3D Object Detection (on nuScenes validation)

|        Model         | Modality | mAP  | NDS  | Checkpoint  |
| :------------------: | :------: | :--: | :--: | :---------: |
|    [BEVFusion](configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml)       |   C+L    | 68.52 | 71.38 | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |
| [Camera-Only Baseline](configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml) |    C     | 35.56 | 41.21 | [Link](https://www.dropbox.com/scl/fi/pxfaz1nc07qa2twlatzkz/camera-only-det.pth?rlkey=f5do81fawie0ssbg9uhrm6p30&dl=1) |
| [LiDAR-Only Baseline](configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml)  |    L     | 64.68 | 69.28 | [Link](https://www.dropbox.com/scl/fi/b1zvgrg9ucmv0wtx6pari/lidar-only-det.pth?rlkey=fw73bmdh57jxtudw6osloywah&dl=1) |

*Note*: The camera-only object detection baseline is a variant of BEVDet-Tiny with a much heavier view transformer and other differences in hyperparameters. Thanks to our [efficient BEV pooling](mmdet3d/ops/bev_pool) operator, this model runs fast and has higher mAP than BEVDet-Tiny under the same input resolution. Please refer to [BEVDet repo](https://github.com/HuangJunjie2017/BEVDet) for the original BEVDet-Tiny implementation. The LiDAR-only baseline is TransFusion-L.

### BEV Map Segmentation (on nuScenes validation)

|        Model         | Modality | mIoU | Checkpoint  |
| :------------------: | :------: | :--: | :---------: |
| [BEVFusion](configs/nuscenes/seg/fusion-bev256d2-lss.yaml)       |   C+L    | 62.95 | [Link](https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9&dl=1) |
| [Camera-Only Baseline](configs/nuscenes/seg/camera-bev256d2.yaml) |    C     | 57.09 | [Link](https://www.dropbox.com/scl/fi/cwpcu80n0shmwraegi6z4/camera-only-seg.pth?rlkey=l60kdaz19fq3gwocsjk09e60z&dl=1) |
| [LiDAR-Only Baseline](configs/nuscenes/seg/lidar-centerpoint-bev128.yaml)  |    L     | 48.56 | [Link](https://www.dropbox.com/scl/fi/mi3w6uxvytdre9i42r9k7/lidar-only-seg.pth?rlkey=rve7hx80u3en1gfoi7tjucl72&dl=1) |

## Usage

### Clone

```shell
git clone https://github.com/Rean-Schwarze/bevfusion.git
```

如果 clone 的是本仓库代码，代码文件修改和新增部分可以跳过（仅做检测推理的话）

### Prerequisites

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)

#### AutoDL version

省钱小技巧：先随便选**一个** GPU，把依赖装好，数据解压、转换好之后，再升配置（

install openmpi

```shell
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.gz
mkdir openmpi
gunzip -c openmpi-4.0.4.tar.gz | tar xf - \
    && cd openmpi-4.0.4 \
    && ./configure --prefix=/root/autodl-tmp/openmpi/ --with-cuda \
    && make all install
```

edit "/etc/profile"

```shell
export PATH=/root/autodl-tmp/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/root/autodl-tmp/openmpi/lib:$LD_LIBRARY_PATH
```

restart

```shell
source /etc/profile
```

update **apt-get**

```shell
sudo apt-get update
sudo apt-get upgrade
```

other dependencies

```shell
# 安装mpi4py时依赖openmpi,不然会报错fatal error: mpi.h
sudo apt-get install wget libgl1-mesa-glx libglib2.0-0 openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y
```

```shell
pip install Pillow==8.4.0 tqdm torchpack nuscenes-devkit mpi4py==3.0.3 numba==0.48.0 setuptools==56.1.0 ninja==1.11.1 numpy==1.23.4 opencv-python==4.8.0.74 opencv-python-headless==4.8.0.74 yapf==0.40.1
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.20.0
```

time cost: ~30min

#### Other

1. 将 mmdet3d/ops/spconv/src/indice_cuda.cu 文件里面所有的 4096 改为 256
2. 算力更改：setup.py 文件中第 22 行左右，只保留一行 "-gencode=arch=compute_86,code=sm_86"

- 参数 86 就是自己显卡的算力根据实际修改，[显卡算力查询](https://developer.nvidia.com/cuda-gpus)

![](https://rean-blog-bucket.oss-cn-guangzhou.aliyuncs.com/assets/img/1729164095080-b2f06e00-cd76-410a-8c39-aaa1752aa387.png)

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

time cost: ~10min

We also provide a [Dockerfile](docker/Dockerfile) to ease environment setup. To get started with docker, please make sure that `nvidia-docker` is installed on your machine. After that, please execute the following command to build the docker image:

```bash
cd docker && docker build . -t bevfusion
```

We can then run the docker with the following command:

```bash
nvidia-docker run -it -v `pwd`/../data:/dataset --shm-size 16g bevfusion /bin/bash
```

We recommend the users to run data preparation (instructions are available in the next section) outside the docker if possible. Note that the dataset directory should be an absolute path. Within the docker, please run the following command to clone our repo and install custom CUDA extensions:

```bash
cd home && git clone https://github.com/mit-han-lab/bevfusion && cd bevfusion
python setup.py develop
```

You can then create a symbolic link `data` to the `/dataset` directory in the docker.

### Data Preparation

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. Please remember to download both detection dataset and the map extension (for BEV map segmentation). After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

Note: Please ensure that your device has at least 500GB of free space if you use the **full** dataset of nuScenes.

##### AutoDL version

tip: 使用 pigz 可加快解压

```shell
sudo apt install pigz
mkdir data
mkdir data/nuscenes
mkdir data/nuscenes/maps
unzip -d /root/autodl-tmp/bevfusion/data/nuscenes/maps /root/autodl-pub/nuScenes/Mapexpansion/nuScenes-map-expansion-v1.3.zip
tar --use-compress-program=pigz -xvpf /root/autodl-pub/nuScenes/Fulldatasetv1.0/Test/v1.0-test_blobs.tgz -C /root/autodl-tmp/bevfusion/data/nuscenes
tar --use-compress-program=pigz -xvpf /root/autodl-pub/nuScenes/Fulldatasetv1.0/Test/v1.0-test_meta.tgz -C /root/autodl-tmp/bevfusion/data/nuscenes
cd /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
for tar in *.tgz;  do tar --use-compress-program=pigz -xvpf $tar -C /root/autodl-tmp/bevfusion/data/nuscenes; done
export PYTHONPATH="/root/autodl-tmp/bevfusion":$PYTHONPATH
```

time cost

```shell
decompress test_blob: ~13min (not using pigz) ~9min (using pigz)
decompress full dataset: ~1h (using pigz)
```

##### Data convert

modify "mmdet3d/ops/**init**.py"

```shell
# from .feature_decorator import feature_decorator
# 把上面这行注释掉
```

更改 tools/data_converter/nuscenes_converter.py 中第 95~100 行，如下图

![](https://rean-blog-bucket.oss-cn-guangzhou.aliyuncs.com/assets/img/1729338664028-699ff04a-1f5a-4a79-a05a-ce02368f224a.png)

```shell
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

time cost

```
create pkl: ~3h36min
create database: ~2h20min
```

After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

### Evaluation

在 AutoDL 中使用，修改"tools/test.py"：

```python
    # dist.init()
    # torch.cuda.set_device(dist.local_rank())

    # init distributed env first, since logger depends on the dist info.
    # distributed = False
    cfg.dist_params = dict(backend='nccl')
    print("args.launcher", args.launcher)    
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
```

在 "tools" 中新增 "dist_test.sh":

```python
#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox --out det.pkl
```

终端中运行：

```python
sudo chmod +777 ./tools/dist_test.sh
```

------

We also provide instructions for evaluating our pretrained models. Please download the checkpoints using the following script: 

First edit "tools/doload_pretrained.sh" to:

```bash
mkdir -p pretrained && \
cd pretrained && \
wget -O bevfusion-det.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/bevfusion-det.pth && \
wget -O bevfusion-seg.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/bevfusion-seg.pth && \
wget -O lidar-only-det.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained/lidar-only-det.pth && \
wget -O lidar-only-seg.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained/lidar-only-seg.pth && \
wget -O camera-only-det.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/camera-only-det.pth && \
wget -O camera-only-seg.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/camera-only-seg.pth && \
wget -O swint-nuimages-pretrained.pth https://hanlab18.mit.edu/projects/bevfusion/files/pretrained_updated/swint-nuimages-pretrained.pth
```

if you use AutoDL, you can enable turbo:

```bash
source /etc/network_turbo
```

then run:

```bash
./tools/download_pretrained.sh
```

Then, you will be able to run:

```bash
torchpack dist-run -np [number of gpus] python tools/test.py [config file path] pretrained/[checkpoint name].pth --eval [evaluation type]
```

For example, if you want to evaluate the detection variant of BEVFusion, you can try:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
```

AutoDL version:

其中最后的数字 2 是 GPU 个数，按需要改

```bash
./tools/dist_test.sh ./configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml ./pretrained/bevfusion-det.pth 2
```

time cost: ~15min (2 3080x2 GPU)

注意：不能直接进行分割的推理，分割和推理需要的转换的数据集不同（很无语好吧

1. 修改 "bevfusion/tools/data_converter/nuscenes_converter.py"

![](https://rean-blog-bucket.oss-cn-guangzhou.aliyuncs.com/assets/img/1729415941563-c61cc23d-00df-45ac-894b-75250e059ead.webp)

```python
        location = nusc.get(
            "log", nusc.get("scene", sample["scene_token"])["log_token"]
        )["location"]
        
        "location": location,
```

2. 修改 "bevfusion/mmdet3d/datasets/nuscenes_dataset.py"

![](https://rean-blog-bucket.oss-cn-guangzhou.aliyuncs.com/assets/img/1729416004252-f37e14d6-ab1f-41dd-9f89-d6c52c02b679.webp)

3. 重新进行数据集转换

While for the segmentation variant of BEVFusion, this command will be helpful:

```bash
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml pretrained/bevfusion-seg.pth --eval map
```

### Common Errors

#### ImportError: cannot import name 'feature_decorator_ext' from partially initialized module 'mmdet3d.ops.feature_decorator' (most likely due to a circular import) (/data/bevfusion/mmdet3d/ops/feature_decorator/__init__.py)

1. setup.py :

Add the below.

```python
make_cuda_ext(
    name="feature_decorator_ext",
    module="mmdet3d.ops.feature_decorator",
    sources=["src/feature_decorator.cpp"],
    sources_cuda=["src/feature_decorator_cuda.cu"],
)
```

and run

```bash
python setup.py develop
```

2. mmdet3d\ops\feature_decorator\src\feature_decorator.cpp :

Delete the below.

```cpp
static auto registry =
torch::RegisterOperators("feature_decorator_ext::feature_decorator_forward", &feature_decorator_forward);
```

#### ModuleNotFoundError: No module named 'flash_attn'

Comment "autodl-tmp/bevfusion/mmdet3d/models/backbones/__init__.py"

```python
# from .radar_encoder import *
```

#### File "/root/bevfusion/mmdet3d/datasets/pipelines/loading.py", line 294, in __call__ location = data["location"] KeyError: 'location'

看上面

### Training

We provide instructions to reproduce our results on nuScenes.

For example, if you want to train the camera-only variant for object detection, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For camera-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/camera-bev256d2.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

For LiDAR-only detector, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

For LiDAR-only BEV segmentation model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/lidar-centerpoint-bev128.yaml
```

For BEVFusion detection model, please run:
```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth 
```

For BEVFusion segmentation model, please run:
```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

Note: please run `tools/test.py` separately after training to get the final evaluation metrics.

## Deployment on TensorRT
[CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion): Best practice for TensorRT, which provides INT8 acceleration solutions and achieves 25fps on ORIN.

## FAQs

Q: Can we directly use the info files prepared by mmdetection3d?

A: We recommend re-generating the info files using this codebase since we forked mmdetection3d before their [coordinate system refactoring](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/changelog.md).

## Acknowledgements

BEVFusion is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). It is also greatly inspired by the following outstanding contributions to the open-source community: [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [BEVDet](https://github.com/HuangJunjie2017/BEVDet), [TransFusion](https://github.com/XuyangBai/TransFusion), [CenterPoint](https://github.com/tianweiy/CenterPoint), [MVP](https://github.com/tianweiy/MVP), [FUTR3D](https://arxiv.org/abs/2203.10642), [CVT](https://github.com/bradyz/cross_view_transformers) and [DETR3D](https://github.com/WangYueFt/detr3d). 

Please also check out related papers in the camera-only 3D perception community such as [BEVDet4D](https://arxiv.org/abs/2203.17054), [BEVerse](https://arxiv.org/abs/2205.09743), [BEVFormer](https://arxiv.org/abs/2203.17270), [M2BEV](https://arxiv.org/abs/2204.05088), [PETR](https://arxiv.org/abs/2203.05625) and [PETRv2](https://arxiv.org/abs/2206.01256), which might be interesting future extensions to BEVFusion.


## Citation

If BEVFusion is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```bibtex
@inproceedings{liu2022bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```
