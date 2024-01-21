# I3D (RGB + Flow)
<figure>
  <img src="../../_assets/i3d.png" width="300" />
</figure>

<!-- TODO: add commit message when it changed -->
!!! note "PWC-Net is deprecated"

    The default behavior has changed in the recent version.
    Now, the optical flow is extracted with [RAFT](https://v-iashin.github.io/video_features/models/raft) instead of PWC-Net (deprecated).

The _Inflated 3D ([I3D](https://arxiv.org/abs/1705.07750))_ features are extracted using
a pre-trained model on [Kinetics 400](https://deepmind.com/research/open-source/kinetics).
Here, the features are extracted from the second-to-the-last layer of I3D, before summing them up.
Therefore, it outputs two tensors with 1024-d features: for RGB and flow streams.
By default, it expects to input 64 RGB and flow frames (`224x224`) which spans 2.56 seconds of the video recorded at 25 fps.
In the default case, the features will be of size `Tv x 1024` where `Tv = duration / 2.56`.

Please note, this implementation uses [RAFT](https://arxiv.org/abs/2003.12039) optical flow extraction instead of the TV-L1 algorithm,
which was used in the original I3D paper as TV-L1 hampers the speed.
Yet, it might possibly lead to worse peformance. Our tests show that the performance is reasonable.
You may check if the predicted distribution satisfies your requirements for an application. To get the predictions that were made by the classification head, providing the `--show_pred` flag.

---
## Supported Arguments

<!-- the <div> makes columns wider -->
| <div style="width: 12em">Argument</div> | <div style="width: 8em">Default</div> | Description                                                                                                                                                                      |
| --------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stack_size`                            | `64`                                  | The number of frames from which to extract features (or window size).                                                                                                            |
| `step_size`                             | `64`                                  | The number of frames to step before extracting the next features.                                                                                                                |
| `streams`                               | `null`                                | I3D is a two-stream network. By default (`null` or omitted) both RGB and flow streams are used. To use RGB- or flow-only models use `rgb` or `flow`.                             |
| `flow_type`                             | `raft`                                 | By default, the flow-features of I3D will be calculated using optical from calculated with RAFT (originally with TV-L1).                    |
| `extraction_fps`                        | `null`                                | If specified (e.g. as `5`), the video will be re-encoded to the `extraction_fps` fps. Leave unspecified or `null` to skip re-encoding.                                           |
| `device`                                | `"cuda:0"`                            | The device specification. It follows the PyTorch style. Use `"cuda:3"` for the 4th GPU on the machine or `"cpu"` for CPU-only.                                                   |
| `video_paths`                           | `null`                                | A list of videos for feature extraction. E.g. `"[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"` or just one path `"./sample/v_GGSY1Qvo990.mp4"`.                      |
| `file_with_video_paths`                 | `null`                                | A path to a text file with video paths (one path per line). Hint: given a folder `./dataset` with `.mp4` files one could use: `find ./dataset -name "*mp4" > ./video_paths.txt`. |
| `on_extraction`                         | `print`                               | If `print`, the features are printed to the terminal. If `save_numpy` or `save_pickle`, the features are saved to either `.npy` file or `.pkl`.                                  |
| `output_path`                           | `"./output"`                          | A path to a folder for storing the extracted features (if `on_extraction` is either `save_numpy` or `save_pickle`).                                                              |
| `keep_tmp_files`                        | `false`                               | If `true`, the reencoded videos will be kept in `tmp_path`.                                                                                                                      |
| `tmp_path`                              | `"./tmp"`                             | A path to a folder for storing temporal files (e.g. reencoded videos).                                                                                                           |
| `show_pred`                             | `false`                               | If `true`, the script will print the predictions of the model on a down-stream task. It is useful for debugging.                                                                 |

---

## Set up the Environment for I3D
Check if you have a correct conda environment installed
```bash
# it will create a new conda environment called 'video_features' on your machine
conda env create -f conda_env.yml
```

---

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LKoytZmNxtC-EuCp7pHDM6sFvK1XdwlW?usp=sharing)

Activate the environment
```bash
conda activate video_features
```

and extract features from `./sample/v_GGSY1Qvo990.mp4` video and show the predicted classes
```bash
python main.py \
    feature_type=i3d \
    device="cuda:0" \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]" \
    show_pred=true
```

---

## Examples
Activate the environment
```bash
conda activate video_features
```

The following will extract I3D features for sample videos.
The features are going to be extracted with the default parameters.
```bash
python main.py \
    feature_type=i3d \
    device="cuda:0" \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```

The video paths can be specified as a `.txt` file with paths
```bash
python main.py \
    feature_type=i3d \
    device="cuda:0" \
    file_with_video_paths=./sample/sample_video_paths.txt
```
It is also possible to extract features from either `rgb` or `flow` modalities individually (`--streams`)
and, therefore, increasing the speed
```bash
python main.py \
    feature_type=i3d \
    streams=flow \
    device="cuda:0" \
    file_with_video_paths=./sample/sample_video_paths.txt
```

The features can be saved as numpy arrays by specifying `--on_extraction save_numpy` or `save_pickle`.
By default, it will create a folder `./output` and will store features there
```bash
python main.py \
    feature_type=i3d \
    device="cuda:0" \
    on_extraction=save_numpy \
    file_with_video_paths=./sample/sample_video_paths.txt
```
You can change the output folder using `--output_path` argument.

Also, you may want to try to change I3D window and step sizes
```bash
python main.py \
    feature_type=i3d \
    device="cuda:0" \
    stack_size=24 \
    step_size=24 \
    file_with_video_paths=./sample/sample_video_paths.txt
```

By default, the frames are extracted according to the original fps of a video. If you would like to extract frames at a certain fps, specify `--extraction_fps` argument.
```bash
python main.py \
    feature_type=i3d \
    device="cuda:0" \
    extraction_fps=25 \
    stack_size=24 \
    step_size=24 \
    file_with_video_paths=./sample/sample_video_paths.txt
```
A fun note, the time span of the I3D features in the last example will match the time span of VGGish features
with default parameters (24/25 = 0.96).

If `--keep_tmp_files` is specified, it keeps them in `--tmp_path` which is `./tmp` by default.
Be careful with the `--keep_tmp_files` argument when playing with `--extraction_fps` as it may mess up the
frames you extracted before in the same folder.

---

## Credits
1. The [Official RAFT implementation (esp. `./demo.py`)](https://github.com/princeton-vl/RAFT/tree/25eb2ac723c36865c636c9d1f497af8023981868).
2. [A port of I3D weights from TensorFlow to PyTorch](https://github.com/hassony2/kinetics_i3d_pytorch)
3. The I3D paper: [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

---

## License
The wrapping code is MIT and the port of I3D weights from TensorFlow to PyTorch. RAFT [BSD 3-Clause](https://github.com/princeton-vl/RAFT/blob/25eb2ac723c36865c636c9d1f497af8023981868/LICENSE).
