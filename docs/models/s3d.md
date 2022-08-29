# R(2+1)D (RGB-only)

<figure>
  <img src="../../_assets/s3d.png" width="300" />
</figure>

The S3D action recognition model was originally introduced in
[Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification](https://arxiv.org/abs/1712.04851).
We support the PyTorch weights for [Kinetics 400](https://deepmind.com/research/open-source/kinetics) provided by
[github.com/kylemin/S3D](https://github.com/kylemin/S3D/commit/098fd72a00186574a316af6af97957c6d7be80a0).
According to the model card, with these weights, the model should achieve 72.08% top-1 accuracy (top5: 90.35%)
on the Kinetics 400 validation set.

How the model was pre-trained?
My best educated guess is that the model was trained on densely sampled 64-frame `224 x 224` stacks
that were randomly trimmed and cropped from 25 fps `256 x 256` video clips (<= 10 sec).
Therefore, to extract features (`Tv x 1024`), we resize the input video such that `min(H, W) = 224` (?)
and take the center crop to make it `224 x 224`.
By default, the feature extractor will split the input video into 64-stack frames (2.56 sec) with no overlap
as it is during the pre-training and will do a forward pass on each of them.
This should be similar to I3D behavior.
For instance, given an ~18-second 25 fps video, the features will be of size `7 x 1024`.
Specify, `step_size`, `extraction_fps`, and `stack_size` to change the default behavior.

What is extracted exactly?
The inputs to the classification head (see `S3D.fc` and `S3D.forward`) that were average-pooled
across the time dimension.


---

## Set up the Environment for S3D
Setup `conda` environment. Requirements are in file `conda_env_torch_zoo.yml`
```bash
# it will create a new conda environment called 'torch_zoo' on your machine
conda env create -f conda_env_torch_zoo.yml
```

---

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HUlYcOJf_dArOcAaR9jaQHuM5CAZiNZc?usp=sharing)

Activate the environment
```bash
conda activate torch_zoo
```

and extract features from the `./sample/v_GGSY1Qvo990.mp4` video and show the predicted classes
```bash
python main.py \
    feature_type=s3d \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]" \
    show_pred=true
```

See the [config file](https://github.com/v-iashin/video_features/blob/master/configs/s3d.yml) for
ther supported parameters.


---

## Credits
1. The [kylemin/S3D](https://github.com/kylemin/S3D/commit/098fd72a00186574a316af6af97957c6d7be80a0) implementation.
2. The S3D paper: [Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification](https://arxiv.org/abs/1712.04851).

---

## License
MIT
