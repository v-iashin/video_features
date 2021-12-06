# VGGish

<figure>
  <img src="../../_assets/vggish.png" width="300" />
</figure>

The [VGGish](https://research.google/pubs/pub45611/) feature extraction relies on the [PyTorch implementation](https://github.com/harritaylor/torchvggish) by [harritaylor](https://github.com/harritaylor) built to replicate the procedure provided in the [TensorFlow repository](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish). The difference in values between the PyTorch and Tensorflow implementation is negligible (see also [# difference in values](#difference-between-tensorflow-and-pytorch-implementations)).

The VGGish model was pre-trained on [AudioSet](https://research.google.com/audioset/). The extracted features are from pre-classification layer after activation. The feature tensor will be 128-d and correspond to 0.96 sec of the original video. Interestingly, this might be represented as 24 frames of a 25 fps video. Therefore, you should expect `Ta x 128` features, where `Ta = duration / 0.96`.

The extraction of VGGish features is implemeted as a wrapper of the TensorFlow implementation. See [Credits](#credits).

---

## Set up the Environment for VGGish
Setup `conda` environment. Requirements are in file `conda_env_torch_zoo.yml`
```bash
# it will create a new conda environment called 'torch_zoo' on your machine
conda env create -f conda_env_torch_zoo.yml
```

---

## Minimal Working Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r_8OnmwXKwmH0n4RxBfuICVBgpbJt_Fs?usp=sharing)

Activate the environment
```bash
conda activate torch_zoo
```

and extract features from the `./sample/v_GGSY1Qvo990.mp4` video
```bash
python main.py \
    feature_type=vggish \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]"
```

---

## Example

The video paths can be specified as a `.txt` file with paths.
```bash
python main.py \
    feature_type=vggish \
    device_ids="[0, 2]" \
    file_with_video_paths=./sample/sample_video_paths.txt
```
The features can be saved as numpy arrays by specifying `--on_extraction save_numpy` or `save_pickle`. By default, it will create a folder `./output` and will store features there (you can change the output folder using `--output_path`)
```bash
python main.py \
    feature_type=vggish \
    device_ids="[0, 2]" \
    on_extraction=save_numpy \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```

---

## Difference between Tensorflow and PyTorch implementations

```
python main.py \
    feature_type=vggish \
    on_extraction=save_numpy \
    file_with_video_paths=./sample/sample_video_paths.txt

TF (./sample/v_GGSY1Qvo990.mp4):
[[0.         0.04247099 0.09079538 ... 0.         0.18485409 0.        ]
 [0.         0.         0.         ... 0.         0.5720243  0.5475726 ]
 [0.         0.00705254 0.15173683 ... 0.         0.33540994 0.10572422]
 ...
 [0.         0.         0.36020872 ... 0.         0.08559107 0.00870359]
 [0.         0.21485361 0.16507196 ... 0.         0.         0.        ]
 [0.         0.31638345 0.         ... 0.         0.         0.        ]]
max: 2.31246495; mean: 0.13741589; min: 0.00000000

PyTorch (./sample/v_GGSY1Qvo990.mp4):
[[0.         0.04247095 0.09079528 ... 0.         0.18485469 0.        ]
 [0.         0.         0.         ... 0.         0.5720252  0.5475726 ]
 [0.         0.0070536  0.1517372  ... 0.         0.33541012 0.10572463]
 ...
 [0.         0.         0.36020786 ... 0.         0.08559084 0.00870359]
 [0.         0.21485506 0.16507116 ... 0.         0.         0.        ]
 [0.         0.31638315 0.         ... 0.         0.         0.        ]]
max: 2.31246495; mean: 0.13741589; min: 0.00000000

(PyTorch - TensorFlow).abs()
tensor([[0.0000e+00, 4.4703e-08, 1.0431e-07,  ..., 0.0000e+00, 5.9605e-07,
         0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00, 8.9407e-07,
         0.0000e+00],
        [0.0000e+00, 1.0580e-06, 3.7253e-07,  ..., 0.0000e+00, 1.7881e-07,
         4.1723e-07],
        ...,
        [0.0000e+00, 0.0000e+00, 8.6427e-07,  ..., 0.0000e+00, 2.3097e-07,
         0.0000e+00],
        [0.0000e+00, 1.4454e-06, 8.0466e-07,  ..., 0.0000e+00, 0.0000e+00,
         0.0000e+00],
        [0.0000e+00, 2.9802e-07, 0.0000e+00,  ..., 0.0000e+00, 0.0000e+00,
         0.0000e+00]])
max: 4.0531e-06; mean: 2.2185e-07; min: 0.00000000
```

---

## Credits
1. The [PyTorch implementation of vggish](https://github.com/harritaylor/torchvggish/tree/f70241ba).
2. The VGGish paper: [CNN Architectures for Large-Scale Audio Classification](https://arxiv.org/abs/1609.09430).

---

## License
The wrapping code is under MIT but the `vggish` implementation complies with the `harritaylor/torchvggish` (same as tensorflow) license which is [Apache-2.0](https://github.com/harritaylor/torchvggish/blob/master/LICENSE).
