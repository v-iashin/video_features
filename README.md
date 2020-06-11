# Multi-GPU Extraction of Video Features

This is a PyTorch module that does a feature extraction in parallel on any number of GPUs. So far, I3D and VGGish features are supported.


## I3D

Please note, this implementation uses PWC-Net instead of the TVL1 algorithm, which was used in the original I3D paper as PWC Net is much faster. Yet, one may create a Pull Request implementing TVL1 as an option to form optical flow frames.

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
1. An implementation of PWC-Net in PyTorch: https://github.com/sniklaus/pytorch-pwc
2. A port of I3D weights from TensorFlow to PyTorch: https://github.com/hassony2/kinetics_i3d_pytorch
3. The I3D paper: [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

### License 
All is MIT except for PWC Net implementation in I3D. Please read the PWC implementation [License](https://github.com/sniklaus/pytorch-pwc) (Last time I checked it was _GPL-3.0_).

## VGGish

The extraction of VGGish features is implemeted as a wrapper of the TensorFlow implementation.

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
See `python main.py --help` for more arguments and I3D examples

### Credits
1. The [TensorFlow implementation](https://github.com/tensorflow/models/tree/0b3a8abf095cb8866ca74c2e118c1894c0e6f947/research/audioset/vggish). 
2. The VGGish paper: [CNN Architectures for Large-Scale Audio Classification](https://arxiv.org/abs/1609.09430).

### License 
My code (this wrapping) is under MIT but the tf implementation complies with the `tensorflow` license which is [Apache-2.0](https://github.com/tensorflow/models/blob/master/LICENSE).