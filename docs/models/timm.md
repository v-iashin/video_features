# timm

`video_features` ❤️ [timm](https://huggingface.co/docs/timm/index).
We support all the models from the `timm` library (technically, for those where you can specify `pretrained=True`).

For details, see the [timm docs](https://huggingface.co/docs/timm/index) and,
specifically [model summaries](https://huggingface.co/docs/timm/models) and
[model benchmark results](https://huggingface.co/docs/timm/results).

## Supported Arguments
<!-- the <div> makes columns wider -->
| <div style="width: 12em">Argument</div> | <div style="width: 8em">Default</div> | Description                                                                                                                                                                      |
| --------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`                            | `null`                            | Any model from `timm.list_pretrained()`, e.g. `efficientnet_b0` or `efficientnet_b0.ra_in1k`.                                                                                     |
| `batch_size`                            | `1`                                   | You may speed up extraction of features by increasing the batch size as much as your GPU permits.                                                                                |
| `extraction_fps`                        | `null`                                | If specified (e.g. as `5`), the video will be re-encoded to the `extraction_fps` fps. Leave unspecified or `null` to skip re-encoding.                                           |
| `device`                                | `"cuda:0"`                            | The device specification. It follows the PyTorch style. Use `"cuda:3"` for the 4th GPU on the machine or `"cpu"` for CPU-only.                                                   |
| `video_paths`                           | `null`                                | A list of videos for feature extraction. E.g. `"[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"` or just one path `"./sample/v_GGSY1Qvo990.mp4"`.                      |
| `file_with_video_paths`                 | `null`                                | A path to a text file with video paths (one path per line). Hint: given a folder `./dataset` with `.mp4` files one could use: `find ./dataset -name "*mp4" > ./video_paths.txt`. |
| `on_extraction`                         | `print`                               | If `print`, the features are printed to the terminal. If `save_numpy` or `save_pickle`, the features are saved to either `.npy` file or `.pkl`.                                  |
| `output_path`                           | `"./output"`                          | A path to a folder for storing the extracted features (if `on_extraction` is either `save_numpy` or `save_pickle`).                                                              |
| `keep_tmp_files`                        | `false`                               | If `true`, the reencoded videos will be kept in `tmp_path`.                                                                                                                      |
| `tmp_path`                              | `"./tmp"`                             | A path to a folder for storing temporal files (e.g. reencoded videos).                                                                                                           |
| `show_pred`                             | `false`                               | If `true`, the script will print the predictions of the model on a down-stream task. It is useful for debugging. This flag is only supported for the models that were trained on ImageNet 1K and 21K.                                                                  |


## Examples

```bash
python main.py \
    feature_type=timm \
    model_name=efficientnet_b0 \
    device="cuda:0" \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```

If you want to specify particular weights, you can do it with `model_name` argument, as you'd do with `timm`,
e.g.
```bash
python main.py \
    feature_type=timm \
    model_name=efficientnet_b0.ra_in1k \
    device="cuda:0" \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]"
```

If you'd like to check the model's outputs on a downstream task (ImageNet 1K or 21K), you can use `show_pred` argument.
```bash
python main.py \
    feature_type=timm \
    model_name=swin_small_patch4_window7_224.ms_in22k \
    device="cuda:0" \
    extraction_fps=1 \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]" \
    show_pred=true
#   Logits | Prob. | Label
#   12.029 | 0.456 | barbell
#   11.676 | 0.321 | weight, free_weight, exercising_weight
#    9.653 | 0.042 | pusher, thruster
#    9.499 | 0.036 | dumbbell
#    8.787 | 0.018 | bench_press

#   Logits | Prob. | Label
#   11.742 | 0.467 | barbell
#   11.233 | 0.281 | weight, free_weight, exercising_weight
#    9.489 | 0.049 | dumbbell
#    8.923 | 0.028 | pusher, thruster
#    8.406 | 0.017 | bench_press

#   Logits | Prob. | Label
#   12.257 | 0.571 | barbell
#   11.391 | 0.240 | weight, free_weight, exercising_weight
#    9.708 | 0.045 | dumbbell
#    9.031 | 0.023 | pusher, thruster
#    8.756 | 0.017 | bench_press

#   Logits | Prob. | Label
#   12.469 | 0.571 | barbell
#   11.655 | 0.253 | weight, free_weight, exercising_weight
#    9.818 | 0.040 | dumbbell
#    9.648 | 0.034 | pusher, thruster
#    8.527 | 0.011 | bench_press

...
```

## Credits
* [timm](https://huggingface.co/docs/timm/index) library

## License
`video_features` is under MIT, the `timm` is under [Apache 2.0](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE).
