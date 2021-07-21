# RAFT
<figure>
  <img src="../../_assets/raft.png" width="300" />
</figure>

[Recurrent All-Pairs Field Transforms for Optical Flow (RAFT)](https://arxiv.org/abs/2003.12039) frames are extracted for every consecutive pair of frames in a video. The implementation follows the [official implementation](https://github.com/princeton-vl/RAFT/tree/25eb2ac723c36865c636c9d1f497af8023981868). RAFT is pre-trained on [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html), fine-tuned on [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), then it is finetuned on [Sintel](http://sintel.is.tue.mpg.de/) or [KITTI-2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) (see the Training Schedule in the Experiments section in the RAFT paper). By default, the frames are extracted using the Sintel model – you may change this behavior in `./models/raft/extract_raft.py`. Also, check out  and [this issue](https://github.com/princeton-vl/RAFT/issues/37) to learn more about the shared models.

The optical flow frames have the same size as the video input or as specified by the resize arguments. We additionally output timesteps in ms for each feature and fps of the video.

---

## Set up the Environment for RAFT
Setup `conda` environment. Requirements for RAFT are similar to the torchvision zoo, which uses `conda_env_torch_zoo.yml`
```bash
# it will create a new conda environment called 'torch_zoo' on your machine
conda env create -f conda_env_torch_zoo.yml
```

---

## Examples
Start by activating the environment
```bash
conda activate torch_zoo
```

A minimal working example:
it will extract RAFT optical flow frames for sample videos using 0th and 2nd devices in parallel.
```bash
python main.py \
    --feature_type raft \
    --device_ids 0 2 \
    --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
Note, if your videos are quite long, have large dimensions and fps, watch your RAM as the frames are stored in the memory until they are saved. Please see other examples how can you overcome this problem.

If you would like to save the frames, use `--on_extraction save_numpy` (or `save_pickle`) – by default, the frames are saved in `./output/` or where `--output_path` specifies. In the case of RAFT, besides frames, it also saves timestamps in ms and the original fps of the video into the same folder with features.
```bash
python main.py \
    --feature_type raft \
    --device_ids 0 2 \
    --on_extraction save_numpy \
    --file_with_video_paths ./sample/sample_video_paths.txt
```
Since extracting flow between two frames is cheap we may increase the extraction speed with batching. Therefore, you can use `--batch_size` argument (defaults to `1`) to do so. _A precaution: make sure to properly test the memory impact of using a specific batch size if you are not sure which kind of videos you have. For instance, you tested the extraction on 16:9 aspect ratio videos but some videos are 16:10 which might give you a mem error. Therefore, I would recommend to tune `--batch_size` on a square video and using the resize arguments (showed later)_
```bash
python main.py \
    --feature_type raft \
    --device_ids 0 2 \
    --batch_size 16 \
    --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
Another way of speeding up the extraction is to resize the input frames. Use `--side_size` to specify the target size of the smallest side (such that `min(W, H) = side_size`) or of the largest side if `--resize_to_larger_edge` is used (such that `max(W, H) = side_size`). The latter might be useful when you are not sure which aspect ratio the videos have.
```bash
python main.py \
    --feature_type raft \
    --device_ids 0 2 \
    --side_size 256 \
    --resize_to_larger_edge \
    --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
If the videos have different fps rate, `--extraction_fps` might be used to specify the target fps of all videos (a video is reencoded and saved to `--tmp_path` folder and deleted if `--keep_tmp_files` wasn't used).
```bash
python main.py \
    --feature_type raft \
    --device_ids 0 2 \
    --extraction_fps 1 \
    --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
Finally, if you would like to test, if the extracted optical flow frames are meaningful or to debug the extraction, use `--show_pred` – it will show the original frame of a video along with the extracted optical flow. (when the window will pop up, use your favorite keys on a keyboard to show the next frame)
```bash
python main.py \
    --feature_type raft \
    --device_ids 0 \
    --show_pred \
    --extraction_fps 5 \
    --video_paths ./sample/v_GGSY1Qvo990.mp4
```

---

## Credits
1. The [Official RAFT implementation (esp. `./demo.py`)](https://github.com/princeton-vl/RAFT/tree/25eb2ac723c36865c636c9d1f497af8023981868).
2. The RAFT paper: [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039).

---

## License
The wrapping code is under MIT, but the RAFT implementation complies with [BSD 3-Clause](https://github.com/princeton-vl/RAFT/blob/25eb2ac723c36865c636c9d1f497af8023981868/LICENSE).
