# VGGish

<figure>
  <img src="../../_assets/vggish.png" width="300" />
</figure>

The [VGGish](https://research.google/pubs/pub45611/) feature extraction mimics the procedure provided in the [TensorFlow repository](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish). Specifically, the VGGish model was pre-trained on [AudioSet](https://research.google.com/audioset/). The extracted features are from pre-classification layer after activation. The feature tensor will be 128-d and correspond to 0.96 sec of the original video. Interestingly, this might be represented as 24 frames of a 25 fps video. Therefore, you should expect `Ta x 128` features, where `Ta = duration / 0.96`.

The extraction of VGGish features is implemeted as a wrapper of the TensorFlow implementation. See [Credits](#credits).

---

## Set up the Environment for VGGish
Setup `conda` environment. Requirements are in file `conda_env_vggish.yml`
```bash
# it will create a new conda environment called 'vggish' on your machine
conda env create -f conda_env_vggish.yml
conda activate vggish
# download the pre-trained VGGish model. The script will put the files in the checkpoint directory
wget https://storage.googleapis.com/audioset/vggish_model.ckpt -P ./models/vggish/checkpoints
```

---

## Example

```bash
python main.py \
    --feature_type vggish \
    --device_ids 0 2 \
    --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```

The video paths can be specified as a `.txt` file with paths
```bash
python main.py \
    --feature_type vggish \
    --device_ids 0 2 \
    --file_with_video_paths ./sample/sample_video_paths.txt
```
The features can be saved as numpy arrays by specifying `--on_extraction save_numpy` or `save_pickle`. By default, it will create a folder `./output` and will store features there (you can change the output folder using `--output_path`)
```bash
python main.py \
    --feature_type vggish \
    --device_ids 0 2 \
    --on_extraction save_numpy \
    --file_with_video_paths ./sample/sample_video_paths.txt
```

---

## Credits
1. The [TensorFlow implementation](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish).
2. The VGGish paper: [CNN Architectures for Large-Scale Audio Classification](https://arxiv.org/abs/1609.09430).

---

## License
The wrapping code is under MIT but the tf implementation complies with the `tensorflow` license which is [Apache-2.0](https://github.com/tensorflow/models/blob/master/LICENSE).
