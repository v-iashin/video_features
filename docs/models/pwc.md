# PWC-Net
<figure>
  <img src="../../_assets/pwc.png" width="300" />
</figure>

[PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371) frames are extracted for every consecutive pair of frames in a video. PWC-Net is pre-trained on [Sintel Flow dataset](http://sintel.is.tue.mpg.de/). The implementation follows [sniklaus/pytorch-pwc@f61389005](https://github.com/sniklaus/pytorch-pwc/tree/f6138900578214ab4e3daef6743b88f7824293be).

---

## Set up the Environment for PWC
Setup `conda` environment. `conda_env_pwc.yml`
```bash
# it will create a new conda environment called 'pwc' on your machine
conda env create -f conda_env_torch_pwc.yml
```

---

## Examples
Start by activating the environment
```bash
conda activate pwc
```

Please see the examples for [`RAFT`](raft.md) optical flow frame extraction. Make sure to replace `--feature_type` argument to `pwc`.

---

## Credits
1. The [PWC-Net paper](https://arxiv.org/abs/1709.02371) and [official implementation](https://github.com/NVlabs/PWC-Net).
2. The [PyTorch implementation used in this repo](https://github.com/sniklaus/pytorch-pwc/tree/f6138900578214ab4e3daef6743b88f7824293be).

---

## License
The wrapping code is under MIT, but PWC Net has [GPL-3.0](https://github.com/sniklaus/pytorch-pwc/blob/f6138900578214ab4e3daef6743b88f7824293be/LICENSE)
