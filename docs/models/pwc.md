# PWC-Net
<figure>
  <img src="../../_assets/pwc.png" width="300" />
</figure>
[PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371) frames are extracted for every consecutive pair of frames in a video. PWC-Net is pre-trained on [Sintel Flow dataset](http://sintel.is.tue.mpg.de/). The implementation follows [sniklaus/pytorch-pwc@f61389005](https://github.com/sniklaus/pytorch-pwc/tree/f6138900578214ab4e3daef6743b88f7824293be).

!!! warning "CUDA 11 and GPUs like RTX 3090 and newer"

    The current environment does not support **CUDA 11** and, therefore, GPUs like **RTX 3090** and newer.
    For details please check this [issue #13](https://github.com/v-iashin/video_features/issues/13)
    If you were able to fix it, please share your workaround.
    If you need an optical flow extractor, you are recommended to use [RAFT](raft.md).

!!! warning "The PWC-Net does NOT support using CPU currently"

    The PWC-Net uses `cupy` module, which makes it difficult to turn to a version that does not use the GPU. However, if you have solution, you may submit a PR.


---

## Set up the Environment for PWC
Setup `conda` environment.
```bash
# it will create a new conda environment called 'pwc' on your machine
conda env create -f conda_env_pwc.yml
```

---

## Quick Start

Activate the environment
```bash
conda activate pwc
```

and extract optical flow from `./sample/v_GGSY1Qvo990.mp4` and show the flow for each frame
```bash
python main.py \
    feature_type=pwc \
    device="cuda:0" \
    show_pred=true \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]"
```
*Note*, if `show_pred=true`, the window with predictions will appear, use any key to show the next frame.
To use `show_pred=true`, a screen must be attached to the machine or X11 forwarding is enabled.

---

## Supported Arguments

<!-- the <div> makes columns wider -->
| <div style="width: 12em">Argument</div> | <div style="width: 8em">Default</div> | Description                                                                                                                                                                      |
| --------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `batch_size`                            | `1`                                   | You may speed up extraction of features by increasing the batch size as much as your GPU permits.                                                                                |
| `extraction_fps`                        | `null`                                | If specified (e.g. as `5`), the video will be re-encoded to the `extraction_fps` fps. Leave unspecified or `null` to skip re-encoding.                                           |
| `side_size`                             | `null`                                | If resized to the smaller edge (`resize_to_smaller_edge=true`), then `min(W, H) = side_size`, if to the larger: max(W, H), if `null` (None) no resize is performed.              |
| `resize_to_smaller_edge`                | `true`                                | If `false`, the larger edge will be used to be resized to `side_size`.                                                                                                           |
| `device`                                | `"cuda:0"`                            | The device specification. It follows the PyTorch style. Use `"cuda:3"` for the 4th GPU on the machine or `"cpu"` for CPU-only.                                                   |
| `video_paths`                           | `null`                                | A list of videos for feature extraction. E.g. `"[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"` or just one path `"./sample/v_GGSY1Qvo990.mp4"`.                      |
| `file_with_video_paths`                 | `null`                                | A path to a text file with video paths (one path per line). Hint: given a folder `./dataset` with `.mp4` files one could use: `find ./dataset -name "*mp4" > ./video_paths.txt`. |
| `on_extraction`                         | `print`                               | If `print`, the features are printed to the terminal. If `save_numpy` or `save_pickle`, the features are saved to either `.npy` file or `.pkl`.                                  |
| `output_path`                           | `"./output"`                          | A path to a folder for storing the extracted features (if `on_extraction` is either `save_numpy` or `save_pickle`).                                                              |
| `keep_tmp_files`                        | `false`                               | If `true`, the reencoded videos will be kept in `tmp_path`.                                                                                                                      |
| `tmp_path`                              | `"./tmp"`                             | A path to a folder for storing temporal files (e.g. reencoded videos).                                                                                                           |
| `show_pred`                             | `false`                               | If `true`, the script will visualize the optical flow for each pair of RGB frames.                                                                                               |

---

## Examples
Please see the examples for [`RAFT`](raft.md) optical flow frame extraction.
Make sure to replace `--feature_type` argument to `pwc`.

---

## Credits
1. The [PWC-Net paper](https://arxiv.org/abs/1709.02371) and [official implementation](https://github.com/NVlabs/PWC-Net).
2. The [PyTorch implementation used in this repo](https://github.com/sniklaus/pytorch-pwc/tree/f6138900578214ab4e3daef6743b88f7824293be).

---

## License
The wrapping code is under MIT, but PWC Net has [GPL-3.0](https://github.com/sniklaus/pytorch-pwc/blob/f6138900578214ab4e3daef6743b88f7824293be/LICENSE)
