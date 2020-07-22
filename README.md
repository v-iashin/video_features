# Multi-GPU Extraction of Video Features

This is a PyTorch module that does a feature extraction in parallel on any number of GPUs. So far, **I3D**, **R(2+1)D** (RGB-only), and **VGGish** features are supported.

- [Multi-GPU Extraction of Video Features](#multi-gpu-extraction-of-video-features)
  - [I3D](#i3d)
    - [Set up the Environment for I3D](#set-up-the-environment-for-i3d)
    - [Examples](#examples)
    - [Credits](#credits)
    - [License](#license)
  - [R(2+1)D (RGB-only)](#r21d-rgb-only)
    - [Set up the Environment for R(2+1)D](#set-up-the-environment-for-r21d)
    - [Example](#example)
    - [Credits](#credits-1)
    - [License](#license-1)
  - [VGGish](#vggish)
    - [Set up the Environment for VGGish](#set-up-the-environment-for-vggish)
    - [Example](#example-1)
    - [Credits](#credits-2)
    - [License](#license-2)

## I3D
The _Inflated 3D ([I3D](https://arxiv.org/abs/1705.07750))_ features are extracted using a pre-trained model on [Kinetics 400](https://deepmind.com/research/open-source/kinetics). Here, the features are extracted from the second-to-the-last layer of I3D, before summing them up. Therefore, it outputs two tensors with 1024-d features: for RGB and flow streams. By default, it expects to input 64 RGB and flow frames (`224x224`) which spans 2.56 seconds of the video recorded at 25 fps. In the default case, the features will be of size `Tv x 1024` where `Tv = duration / 2.56`.

Please note, this implementation uses [PWC-Net](https://arxiv.org/abs/1709.02371) instead of the TV-L1 algorithm, which was used in the original I3D paper as PWC-Net is much faster. Yet, it might possibly lead to worse peformance. We tested it with PWC-Net flow frames and found that the performance is reasonable. You may test it yourself by providing `--show_kinetics_pred` flag. Also, one may create a Pull Request implementing [TV-L1](https://docs.opencv.org/master/d4/dee/tutorial_optical_flow.html) as an option to form optical flow frames.

### Set up the Environment for I3D
Setup `conda` environment. Requirements are in file `conda_env_i3d.yml`
```bash
# it will create new conda environment called 'i3d' on your machine 
conda env create -f conda_env_i3d.yml
conda activate i3d
```

### Examples
It will extract I3D features for sample videos using 0th and 2nd devices in parallel. The features are going to be extracted with the default parameters. Check out `python main.py --help` for help on available options.
```bash
python main.py --feature_type i3d --device_ids 0 2 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```

The video paths can be specified as a `.txt` file with paths
```bash
python main.py --feature_type i3d --device_ids 0 2 --file_with_video_paths ./sample/sample_video_paths.txt
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

If `--keep_frames` is specified, it keeps them in `--tmp_path` which is `./tmp` by default. Be careful with the `--keep_frames` argument when playing with `--extraction_fps` as it may mess up the frames you extracted before in the same folder.

### Credits
1. [An implementation of PWC-Net in PyTorch](https://github.com/sniklaus/pytorch-pwc)
2. [A port of I3D weights from TensorFlow to PyTorch](https://github.com/hassony2/kinetics_i3d_pytorch)
3. The I3D paper: [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

### License 
The wrapping code is MIT and the port of I3D weights from TensorFlow to PyTorch. However, PWC Net has a different [License](https://github.com/sniklaus/pytorch-pwc) (Last time I checked it was _GPL-3.0_).

## R(2+1)D (RGB-only)
The extraction of an [18-layer R(2+1)D (RGB-only)](https://arxiv.org/abs/1711.11248) network is borrowed from [torchvision models](https://pytorch.org/docs/1.5.0/torchvision/models.html#resnet-2-1-d). Similar to [I3D](#i3d), R(2+1)D is pre-trained on [Kinetics 400](https://deepmind.com/research/open-source/kinetics). The features are extracted from the pre-classification layer of the net. Therefore, it outputs a tensor with 512-d features for each stack. By default, [according to torchvision docs](https://pytorch.org/docs/1.5.0/torchvision/models.html#video-classification), it expects to input a stack of 16 RGB frames (`112x112`), which spans 0.64 seconds of the video recorded at 25 fps. Specify `--step_size` and `--stack_size` to change the default behavior. In the default case, the features will be of size `Tv x 512` where `Tv = duration / 0.64`. The augmentations are similar to the proposed in [torchvision training scripts](https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/references/video_classification/train.py#L154-L159). 

### Set up the Environment for R(2+1)D
Setup `conda` environment. Requirements are in file `conda_env_r21d.yml`
```bash
# it will create new conda environment called 'r21d' on your machine 
conda env create -f conda_env_r21d.yml
conda activate r21d
```

### Example

```bash
python main.py --feature_type r21d_rgb --device_ids 0 2 --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
See `python main.py --help` for more arguments and [I3D Examples](#examples).

### Credits
1. The [TorchVision implementation](https://pytorch.org/docs/1.5.0/torchvision/models.html#video-classification). 
2. The R(2+1)D paper: [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248).

### License
The wrapping code is under MIT, yet, it utilizes `torchvision` library which is under [BSD 3-Clause "New" or "Revised" License](https://github.com/pytorch/vision/blob/master/LICENSE).

## VGGish

The [VGGish](https://research.google/pubs/pub45611/) feature extraction mimics the procedure provided in the [TensorFlow repository](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish). Specifically, the VGGish model was pre-trained on [AudioSet](https://research.google.com/audioset/). The extracted features are from pre-classification layer after activation. The feature tensor will be 128-d and correspond to 0.96 sec of the original video. Interestingly, this might be represented as 24 frames of a 25 fps video. Therefore, you should expect `Ta x 128` features, where `Ta = duration / 0.96`.

The extraction of VGGish features is implemeted as a wrapper of the TensorFlow implementation. See [Credits](#credits).


### Set up the Environment for VGGish
Setup `conda` environment. Requirements are in file `conda_env_vggish.yml`
```bash
# it will create new conda environment called 'vggish' on your machine 
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