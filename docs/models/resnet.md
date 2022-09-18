# ResNet

<figure>
  <img src="../../_assets/resnet.png" width="300" />
</figure>

The [ResNet](https://arxiv.org/abs/1512.03385) features are extracted at each frame of the provided video.
The ResNet is pre-trained on the 1k ImageNet dataset.
We extract features from the pre-classification layer.
The implementation is based on the [torchvision models](https://pytorch.org/docs/1.6.0/torchvision/models.html#classification).
The extracted features are going to be of size `num_frames x 2048`.
We additionally output timesteps in ms for each feature and fps of the video. We use the standard set of augmentations.

---

## Set up the Environment for ResNet
Setup `conda` environment. Requirements are in file `conda_env_torch_zoo.yml`
```bash
# it will create a new conda environment called 'torch_zoo' on your machine
conda env create -f conda_env_torch_zoo.yml
```

---

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17VLdf4abQT2eoMjc6ziJ9UaRaOklTlP0?usp=sharing)

Activate the environment
```bash
conda activate torch_zoo
```

and extract features at 1 fps from `./sample/v_GGSY1Qvo990.mp4` video and show the predicted classes
```bash
python main.py \
    feature_type=resnet \
    model_name=resnet101 \
    extraction_fps=1 \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]" \
    show_pred=true
```

---

## Supported Arguments

<!-- the <div> makes columns wider -->
| <div style="width: 12em">Argument</div> | <div style="width: 8em">Default</div> | Description                                                                                                                                                                      |
| --------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`                            | `resnet50`                            | A variant of ResNet. `resnet18` `resnet34` `resnet50` `resnet101` `resnet151` are supported.                                                                                     |
| `batch_size`                            | `1`                                   | You may speed up extraction of features by increasing the batch size as much as your GPU permits.                                                                                |
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

## Examples
Start by activating the environment
```bash
conda activate torch_zoo
```

It is pretty much the same procedure as with other features.
The example is provided for the ResNet-50 flavour, but we also support ResNet-18,34,101,152.
You can specify the model with the `model_name` parameter
```bash
python main.py \
    feature_type=resnet \
    model_name=resnet50 \
    device="cuda:0" \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```
If you would like to save the features, use `--on_extraction save_numpy` (or `save_pickle`) – by default,
the features are saved in `./output/` or where `--output_path` specifies.
In the case of frame-wise features, besides features, it also saves timestamps in ms and the original fps of
the video into the same folder with features.
```bash
python main.py \
    feature_type=resnet \
    model_name=resnet50 \
    device="cuda:0" \
    on_extraction=save_numpy \
    file_with_video_paths=./sample/sample_video_paths.txt
```
Since these features are so fine-grained and light-weight we may increase the extraction speed with batching.
Therefore, frame-wise features have `--batch_size` argument, which defaults to `1`.
```bash
python main.py \
    feature_type=resnet \
    model_name=resnet50 \
    device="cuda:0" \
    batch_size=128 \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```
If you would like to extract features at a certain fps, add `--extraction_fps` argument
```bash
python main.py \
    feature_type=resnet \
    model_name=resnet50 \
    device="cuda:0" \
    extraction_fps=5 \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```

---

## Credits
1. The [TorchVision implementation](https://pytorch.org/docs/1.6.0/torchvision/models.html#classification).
2. The [ResNet paper](https://arxiv.org/abs/1512.03385)

---

## License
The wrapping code is under MIT, yet, it utilizes `torchvision` library which is
under [BSD 3-Clause "New" or "Revised" License](https://github.com/pytorch/vision/blob/master/LICENSE).
