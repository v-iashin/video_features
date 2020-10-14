# Multi-GPU Extraction of Video Features

This is a PyTorch module that does a feature extraction in parallel on any number of GPUs. So far, **I3D** (RGB + Flow), **R(2+1)D** (RGB-only), and **VGGish** features are supported as well as **ResNet-50** (frame-wise). Now, it also supports optical flow frame extraction using **RAFT** and **PWC-Net**.

- [Multi-GPU Extraction of Video Features](#multi-gpu-extraction-of-video-features)
  - [I3D (RGB + Flow)](#i3d-rgb--flow)
    - [Set up the Environment for I3D](#set-up-the-environment-for-i3d)
    - [Examples](#examples)
    - [Credits](#credits)
    - [License](#license)
  - [R(2+1)D (RGB-only)](#r21d-rgb-only)
    - [Set up the Environment for R(2+1)D](#set-up-the-environment-for-r21d)
    - [Example](#example)
    - [Credits](#credits-1)
    - [License](#license-1)
  - [ResNet-50 (frame-wise)](#resnet-50-frame-wise)
    - [Set up the Environment for ResNet-50](#set-up-the-environment-for-resnet-50)
    - [Examples](#examples-1)
    - [Credits](#credits-2)
    - [License](#license-2)
  - [RAFT (Optical Flow, frame-wise)](#raft-optical-flow-frame-wise)
    - [Set up the Environment for RAFT](#set-up-the-environment-for-raft)
    - [Examples](#examples-2)
    - [Credits](#credits-3)
    - [License](#license-3)
  - [PWC-Net (Optical Flow, frame-wise)](#pwc-net-optical-flow-frame-wise)
    - [Set up the Environment for PWC](#set-up-the-environment-for-pwc)
    - [Examples](#examples-3)
    - [Credits](#credits-4)
    - [License](#license-4)
  - [VGGish](#vggish)
    - [Set up the Environment for VGGish](#set-up-the-environment-for-vggish)
    - [Example](#example-1)
    - [Credits](#credits-5)
    - [License](#license-5)

## I3D (RGB + Flow)
The _Inflated 3D ([I3D](https://arxiv.org/abs/1705.07750))_ features are extracted using a pre-trained model on [Kinetics 400](https://deepmind.com/research/open-source/kinetics). Here, the features are extracted from the second-to-the-last layer of I3D, before summing them up. Therefore, it outputs two tensors with 1024-d features: for RGB and flow streams. By default, it expects to input 64 RGB and flow frames (`224x224`) which spans 2.56 seconds of the video recorded at 25 fps. In the default case, the features will be of size `Tv x 1024` where `Tv = duration / 2.56`.

Please note, this implementation uses either [PWC-Net](https://arxiv.org/abs/1709.02371) (the default) and [RAFT](https://arxiv.org/abs/2003.12039) optical flow extraction instead of the TV-L1 algorithm, which was used in the original I3D paper as it hampers speed. Yet, it might possibly lead to worse peformance. Our tests show that the performance is reasonable. You may test it yourself by providing `--show_pred` flag.

### Set up the Environment for I3D
Depending on whether you would like to use PWC-Net or RAFT for optical flow extraction, you will need to install separate conda environments – `conda_env_pwc.yml` and `conda_env_torch_zoo`, respectively
```bash
# it will create a new conda environment called 'pwc' (or/and `torch_zoo`) on your machine
conda env create -f conda_env_pwc.yml
# or/and
conda env create -f conda_env_torch_zoo.yml
```

### Examples
Start by activating the environment
```bash
conda activate pwc
```

The following will extract I3D features for sample videos using 0th and 2nd devices in parallel. The features are going to be extracted with the default parameters.
```bash
python main.py --feature_type i3d --device_ids 0 2 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```

The video paths can be specified as a `.txt` file with paths
```bash
python main.py --feature_type i3d --device_ids 0 2 --file_with_video_paths ./sample/sample_video_paths.txt
```
It is also possible to extract features from either `rgb` or `flow` modalities individually (`--streams`) and, therefore, increasing the speed
```bash
python main.py --feature_type i3d --streams flow --device_ids 0 2 --file_with_video_paths ./sample/sample_video_paths.txt
```

To extract optical flow frames using RAFT approach, specify `--flow_type raft`. Note that using RAFT will make the extraction slower than with PWC-Net yet visual inspection of extracted flow frames suggests that RAFT has a better quality of the estimated flow
```bash
# make usre to activate the correct environment (`torch_zoo`)
python main.py --feature_type i3d --flow_type raft --device_ids 0 2 --file_with_video_paths ./sample/sample_video_paths.txt
```

The features can be saved as numpy arrays by specifying `--on_extraction save_numpy`. By default, it will create a folder `./output` and will store features there
```bash
python main.py --feature_type i3d --device_ids 0 2 --on_extraction save_numpy --file_with_video_paths ./sample/sample_video_paths.txt
```
You can change the output folder using `--output_path` argument.

Also, you may want to try to change I3D window and step sizes
```bash
python main.py --feature_type i3d --device_ids 0 2 --stack_size 24 --step_size 24 --file_with_video_paths ./sample/sample_video_paths.txt
```

By default, the frames are extracted according to the original fps of a video. If you would like to extract frames at a certain fps, specify `--extraction_fps` argument.
```bash
python main.py --feature_type i3d --device_ids 0 2 --extraction_fps 25 --stack_size 24 --step_size 24 --file_with_video_paths ./sample/sample_video_paths.txt
```
A fun note, the time span of the I3D features in the last example will match the time span of VGGish features with default parameters (24/25 = 0.96).

If `--keep_tmp_files` is specified, it keeps them in `--tmp_path` which is `./tmp` by default. Be careful with the `--keep_tmp_files` argument when playing with `--extraction_fps` as it may mess up the frames you extracted before in the same folder.

### Credits
1. [An implementation of PWC-Net in PyTorch](https://github.com/sniklaus/pytorch-pwc/tree/f6138900578214ab4e3daef6743b88f7824293be)
2. The [Official RAFT implementation (esp. `./demo.py`)](https://github.com/princeton-vl/RAFT/tree/25eb2ac723c36865c636c9d1f497af8023981868).
3. [A port of I3D weights from TensorFlow to PyTorch](https://github.com/hassony2/kinetics_i3d_pytorch)
4. The I3D paper: [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

### License
The wrapping code is MIT and the port of I3D weights from TensorFlow to PyTorch. However, PWC Net (default flow extractor) has [GPL-3.0](https://github.com/sniklaus/pytorch-pwc/blob/f6138900578214ab4e3daef6743b88f7824293be/LICENSE) and RAFT [BSD 3-Clause](https://github.com/princeton-vl/RAFT/blob/25eb2ac723c36865c636c9d1f497af8023981868/LICENSE).

## R(2+1)D (RGB-only)
The extraction of an [18-layer R(2+1)D (RGB-only)](https://arxiv.org/abs/1711.11248) network is borrowed from [torchvision models](https://pytorch.org/docs/1.5.0/torchvision/models.html#resnet-2-1-d). Similar to [I3D](#i3d), R(2+1)D is pre-trained on [Kinetics 400](https://deepmind.com/research/open-source/kinetics). The features are extracted from the pre-classification layer of the net. Therefore, it outputs a tensor with 512-d features for each stack. By default, [according to torchvision docs](https://pytorch.org/docs/1.5.0/torchvision/models.html#video-classification), it expects to input a stack of 16 RGB frames (`112x112`), which spans 0.64 seconds of the video recorded at 25 fps. Specify `--step_size` and `--stack_size` to change the default behavior. In the default case, the features will be of size `Tv x 512` where `Tv = duration / 0.64`. The augmentations are similar to the proposed in [torchvision training scripts](https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/references/video_classification/train.py#L154-L159).

### Set up the Environment for R(2+1)D
Setup `conda` environment. Requirements are in file `conda_env_torch_zoo.yml`
```bash
# it will create a new conda environment called 'torch_zoo' on your machine
conda env create -f conda_env_torch_zoo.yml
```

### Example
Start by activating the environment
```bash
conda activate torch_zoo
```

It will extract R(2+1)d features for sample videos using 0th and 2nd devices in parallel. The features are going to be extracted with the default parameters.
```bash
python main.py --feature_type r21d_rgb --device_ids 0 2 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
See [I3D Examples](#examples). Note, that R(2+1)d only supports RGB stream.

### Credits
1. The [TorchVision implementation](https://pytorch.org/docs/1.5.0/torchvision/models.html#video-classification).
2. The R(2+1)D paper: [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248).

### License
The wrapping code is under MIT, yet, it utilizes `torchvision` library which is under [BSD 3-Clause "New" or "Revised" License](https://github.com/pytorch/vision/blob/master/LICENSE).

## ResNet-50 (frame-wise)

The [ResNet-50](https://arxiv.org/abs/1512.03385) features are extracted frame-wise for a provided video. The ResNet-50 is pre-trained on the 1k ImageNet dataset. We extract features from the pre-classification layer. The implementation is based on the [torchvision models](https://pytorch.org/docs/1.6.0/torchvision/models.html#classification). The extracted features are going to be of size `num_frames x 2048`. We additionally output timesteps in ms for each feature and fps of the video. We use the standard set of augmentations.

### Set up the Environment for ResNet-50
Setup `conda` environment. Requirements are in file `conda_env_torch_zoo.yml`
```bash
# it will create a new conda environment called 'torch_zoo' on your machine
conda env create -f conda_env_torch_zoo.yml
```

### Examples
Start by activating the environment
```bash
conda activate torch_zoo
```

It is pretty much the same procedure as with other features.
```bash
python main.py --feature_type resnet50 --device_ids 0 2 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
If you would like to save the features, use `--on_extraction save_numpy` – by default, the features are saved in `./output/` or where `--output_path` specifies. In the case of frame-wise features, besides features, it also saves timestamps in ms and the original fps of the video into the same folder with features.
```bash
python main.py --feature_type resnet50 --device_ids 0 2 --on_extraction save_numpy --file_with_video_paths ./sample/sample_video_paths.txt
```
Since these features are so fine-grained and light-weight we may increase the extraction speed with batching. Therefore, frame-wise features have `--batch_size` argument, which defaults to `1`.
```bash
python main.py --feature_type resnet50 --device_ids 0 2 --batch_size 128 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
If you would like to extract features at a certain fps, add `--extraction_fps` argument
```bash
python main.py --feature_type resnet50 --device_ids 0 2 --extraction_fps 5 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```

### Credits
1. The [TorchVision implementation](https://pytorch.org/docs/1.6.0/torchvision/models.html#classification).
2. The [ResNet paper](https://arxiv.org/abs/1512.03385)

### License
The wrapping code is under MIT, yet, it utilizes `torchvision` library which is under [BSD 3-Clause "New" or "Revised" License](https://github.com/pytorch/vision/blob/master/LICENSE).

## RAFT (Optical Flow, frame-wise)

[Recurrent All-Pairs Field Transforms for Optical Flow (RAFT)](https://arxiv.org/abs/2003.12039) frames are extracted for every consecutive pair of frames in a video. The implementation follows the [official implementation](https://github.com/princeton-vl/RAFT/tree/25eb2ac723c36865c636c9d1f497af8023981868). RAFT is pre-trained on [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html), fine-tuned on [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), then it is finetuned on [Sintel](http://sintel.is.tue.mpg.de/) or [KITTI-2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) (see the Training Schedule in the Experiments section in the RAFT paper). By default, the frames are extracted using the Sintel model – you may change this behavior in `./models/raft/extract_raft.py`. Also, check out  and [this issue](https://github.com/princeton-vl/RAFT/issues/37) to learn more about the shared models.

The optical flow frames have the same size as the video input or as specified by the resize arguments. We additionally output timesteps in ms for each feature and fps of the video.

### Set up the Environment for RAFT
Setup `conda` environment. Requirements for RAFT are similar to the torchvision zoo, which uses `conda_env_torch_zoo.yml`
```bash
# it will create a new conda environment called 'torch_zoo' on your machine
conda env create -f conda_env_torch_zoo.yml
```

### Examples
Start by activating the environment
```bash
conda activate torch_zoo
```

A minimal working example:
it will extract RAFT optical flow frames for sample videos using 0th and 2nd devices in parallel.
```bash
python main.py --feature_type raft --device_ids 0 2 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
Note, if your videos are quite long, have large dimensions and fps, watch your RAM as the frames are stored in the memory until they are saved. Please see other examples how can you overcome this problem.

If you would like to save the frames, use `--on_extraction save_numpy` – by default, the frames are saved in `./output/` or where `--output_path` specifies. In the case of RAFT, besides frames, it also saves timestamps in ms and the original fps of the video into the same folder with features.
```bash
python main.py --feature_type raft --device_ids 0 2 --on_extraction save_numpy --file_with_video_paths ./sample/sample_video_paths.txt
```
Since extracting flow between two frames is cheap we may increase the extraction speed with batching. Therefore, you can use `--batch_size` argument (defaults to `1`) to do so. _A precaution: make sure to properly test the memory impact of using a specific batch size if you are not sure which kind of videos you have. For instance, you tested the extraction on 16:9 aspect ratio videos but some videos are 16:10 which might give you a mem error. Therefore, I would recommend to tune `--batch_size` on a square video and using the resize arguments (showed later)_
```bash
python main.py --feature_type raft --device_ids 0 2 --batch_size 16 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
Another way of speeding up the extraction is to resize the input frames. Use `--side_size` to specify the target size of the smallest side (such that `min(W, H) = side_size`) or of the largest side if `--resize_to_larger_edge` is used (such that `max(W, H) = side_size`). The latter might be useful when you are not sure which aspect ratio the videos have.
```bash
python main.py --feature_type raft --device_ids 0 2 --side_size 256 --resize_to_larger_edge --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
If the videos have different fps rate, `--extraction_fps` might be used to specify the target fps of all videos (a video is reencoded and saved to `--tmp_path` folder and deleted if `--keep_tmp_files` wasn't used).
```bash
python main.py --feature_type raft --device_ids 0 2 --extraction_fps 1 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
Finally, if you would like to test, if the extracted optical flow frames are meaningful or to debug the extraction, use `--show_pred` – it will show the original frame of a video along with the extracted optical flow. (when the window will pop up, use your favorite keys on a keyboard to show the next frame)
```bash
python main.py --feature_type raft --device_ids 0 --show_pred --extraction_fps 5 --video_paths ./sample/v_GGSY1Qvo990.mp4
```

### Credits
1. The [Official RAFT implementation (esp. `./demo.py`)](https://github.com/princeton-vl/RAFT/tree/25eb2ac723c36865c636c9d1f497af8023981868).
2. The RAFT paper: [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039).

### License
The wrapping code is under MIT, but the RAFT implementation complies with [BSD 3-Clause](https://github.com/princeton-vl/RAFT/blob/25eb2ac723c36865c636c9d1f497af8023981868/LICENSE).

## PWC-Net (Optical Flow, frame-wise)
[PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371) frames are extracted for every consecutive pair of frames in a video. PWC-Net is pre-trained on [Sintel Flow dataset](http://sintel.is.tue.mpg.de/). The implementation follows [sniklaus/pytorch-pwc@f61389005](https://github.com/sniklaus/pytorch-pwc/tree/f6138900578214ab4e3daef6743b88f7824293be).


### Set up the Environment for PWC
Setup `conda` environment. `conda_env_pwc.yml`
```bash
# it will create a new conda environment called 'pwc' on your machine
conda env create -f conda_env_torch_pwc.yml
```

### Examples
Start by activating the environment
```bash
conda activate pwc
```

Please see the examples for `RAFT` optical flow frame extraction. Make sure to replace `--feature_type` argument to `pwc`.

### Credits
1. The [PWC-Net paper](https://arxiv.org/abs/1709.02371) and [official implementation](https://github.com/NVlabs/PWC-Net).
2. The [PyTorch implementation used in this repo](https://github.com/sniklaus/pytorch-pwc/tree/f6138900578214ab4e3daef6743b88f7824293be).

### License
The wrapping code is under MIT, but PWC Net has [GPL-3.0](https://github.com/sniklaus/pytorch-pwc/blob/f6138900578214ab4e3daef6743b88f7824293be/LICENSE)

## VGGish

The [VGGish](https://research.google/pubs/pub45611/) feature extraction mimics the procedure provided in the [TensorFlow repository](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish). Specifically, the VGGish model was pre-trained on [AudioSet](https://research.google.com/audioset/). The extracted features are from pre-classification layer after activation. The feature tensor will be 128-d and correspond to 0.96 sec of the original video. Interestingly, this might be represented as 24 frames of a 25 fps video. Therefore, you should expect `Ta x 128` features, where `Ta = duration / 0.96`.

The extraction of VGGish features is implemeted as a wrapper of the TensorFlow implementation. See [Credits](#credits).


### Set up the Environment for VGGish
Setup `conda` environment. Requirements are in file `conda_env_vggish.yml`
```bash
# it will create a new conda environment called 'vggish' on your machine
conda env create -f conda_env_vggish.yml
conda activate vggish
# download the pre-trained VGGish model. The script will put the files in the checkpoint directory
wget https://storage.googleapis.com/audioset/vggish_model.ckpt -P ./models/vggish/checkpoints
```

### Example

```bash
python main.py --feature_type vggish --device_ids 0 2 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
See `python main.py --help` for more arguments and [I3D Examples](#examples).

### Credits
1. The [TensorFlow implementation](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish).
2. The VGGish paper: [CNN Architectures for Large-Scale Audio Classification](https://arxiv.org/abs/1609.09430).

### License
The wrapping code is under MIT but the tf implementation complies with the `tensorflow` license which is [Apache-2.0](https://github.com/tensorflow/models/blob/master/LICENSE).
