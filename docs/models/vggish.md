# VGGish

<figure>
  <img src="../../_assets/vggish.png" width="300" />
</figure>

The [VGGish](https://research.google/pubs/pub45611/) feature extraction relies on the [PyTorch implementation](https://github.com/harritaylor/torchvggish) by [harritaylor](https://github.com/harritaylor) built to replicate the procedure provided in the [TensorFlow repository](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish). The difference in values between the PyTorch and Tensorflow implementation is negligible (see also [# difference in values](#difference-between-tensorflow-and-pytorch-implementations)).

The VGGish model was pre-trained on [AudioSet](https://research.google.com/audioset/). The extracted features are from pre-classification layer after activation. The feature tensor will be 128-d and correspond to 0.96 sec of the original video. Interestingly, this might be represented as 24 frames of a 25 fps video. Therefore, you should expect `Ta x 128` features, where `Ta = duration / 0.96`.

The extraction of VGGish features is implemeted as a wrapper of the TensorFlow implementation. See [Credits](#credits).

---

## Set up the Environment for VGGish
Setup `conda` environment. Requirements are in file `conda_env.yml`
```bash
# it will create a new conda environment called 'video_features' on your machine
conda env create -f conda_env.yml
```

---

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r_8OnmwXKwmH0n4RxBfuICVBgpbJt_Fs?usp=sharing)

Activate the environment
```bash
conda activate video_features
```

and extract features from the `./sample/v_GGSY1Qvo990.mp4` video
```bash
python main.py \
    feature_type=vggish \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]"
```

---

## Supported Arguments

<!-- the <div> makes columns wider -->
| <div style="width: 12em">Argument</div> | <div style="width: 8em">Default</div> | Description                                                                                                                                                                      |
| --------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `device`                                | `"cuda:0"`                            | The device specification. It follows the PyTorch style. Use `"cuda:3"` for the 4th GPU on the machine or `"cpu"` for CPU-only.                                                   |
| `video_paths`                           | `null`                                | A list of videos for feature extraction. E.g. `"[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"` or just one path `"./sample/v_GGSY1Qvo990.mp4"`.                      |
| `file_with_video_paths`                 | `null`                                | A path to a text file with video paths (one path per line). Hint: given a folder `./dataset` with `.mp4` files one could use: `find ./dataset -name "*mp4" > ./video_paths.txt`. |
| `on_extraction`                         | `print`                               | If `print`, the features are printed to the terminal. If `save_numpy` or `save_pickle`, the features are saved to either `.npy` file or `.pkl`.                                  |
| `output_path`                           | `"./output"`                          | A path to a folder for storing the extracted features (if `on_extraction` is either `save_numpy` or `save_pickle`).                                                              |
| `keep_tmp_files`                        | `false`                               | If `true`, the reencoded videos will be kept in `tmp_path`.                                                                                                                      |
| `tmp_path`                              | `"./tmp"`                             | A path to a folder for storing temporal files (e.g. reencoded videos).                                                                                                           |

---

## Example

The video paths can be specified as a `.txt` file with paths.
```bash
python main.py \
    feature_type=vggish \
    device="cuda:0" \
    file_with_video_paths=./sample/sample_video_paths.txt
```
The features can be saved as numpy arrays by specifying `--on_extraction save_numpy` or `save_pickle`. By default, it will create a folder `./output` and will store features there (you can change the output folder using `--output_path`)
```bash
python main.py \
    feature_type=vggish \
    device="cuda:0" \
    on_extraction=save_numpy \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```

---

## Difference between TensorFlow and PyTorch implementations
VGGish was originally implemented in TensorFlow.
We use the PyTorch implementation by
[harritaylor/torchvggish](https://github.com/harritaylor/torchvggish/tree/f70241ba)
The difference in values between the PyTorch and Tensorflow implementation is negligible.
However, after updating the versions of the dependencies, the values are slightly different.
If you wish to use the old implementation, you can use the conda environment at the `b21f330` commit or earlier.
The following table shows the difference in values.

```
python main.py \
    feature_type=vggish \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]"

Original (./sample/v_GGSY1Qvo990.mp4):
[[0.         0.04247099 0.09079538 ... 0.         0.18485409 0.        ]
 [0.         0.         0.         ... 0.         0.5720243  0.5475726 ]
 [0.         0.00705254 0.15173683 ... 0.         0.33540994 0.10572422]
 ...
 [0.         0.         0.36020872 ... 0.         0.08559107 0.00870359]
 [0.         0.21485361 0.16507196 ... 0.         0.         0.        ]
 [0.         0.31638345 0.         ... 0.         0.         0.        ]]
max: 2.31246495; mean: 0.13741589; min: 0.00000000

b21f330 and ealier (./sample/v_GGSY1Qvo990.mp4):
[[0.         0.04247095 0.09079528 ... 0.         0.18485469 0.        ]
 [0.         0.         0.         ... 0.         0.5720252  0.5475726 ]
 [0.         0.0070536  0.1517372  ... 0.         0.33541012 0.10572463]
 ...
 [0.         0.         0.36020786 ... 0.         0.08559084 0.00870359]
 [0.         0.21485506 0.16507116 ... 0.         0.         0.        ]
 [0.         0.31638315 0.         ... 0.         0.         0.        ]]
max: 2.31246495; mean: 0.13741589; min: 0.00000000

Current (./sample/v_GGSY1Qvo990.mp4):
[[0.         0.0752698  0.12985817 ... 0.         0.18340725 0.00647891]
 [0.         0.         0.         ... 0.         0.5479691  0.6105871 ]
 [0.         0.03563304 0.1507446  ... 0.         0.20983526 0.15856776]
 ...
 [0.         0.         0.3077196  ... 0.         0.08271158 0.03223182]
 [0.         0.15476668 0.25240228 ... 0.         0.         0.        ]
 [0.         0.3711498  0.         ... 0.         0.         0.        ]]
max: 2.41924119; mean: 0.13830526; min: 0.00000000

```

---

## Credits
1. The [PyTorch implementation of vggish](https://github.com/harritaylor/torchvggish/tree/f70241ba).
2. The VGGish paper: [CNN Architectures for Large-Scale Audio Classification](https://arxiv.org/abs/1609.09430).

---

## License
The wrapping code is under MIT but the `vggish` implementation complies with the `harritaylor/torchvggish` (same as tensorflow) license which is [Apache-2.0](https://github.com/harritaylor/torchvggish/blob/master/LICENSE).
