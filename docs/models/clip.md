# CLIP


The [CLIP](https://arxiv.org/abs/2103.00020) features are extracted at each frame of the provided video.
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs.
We use CLIP's official augmentations and extract vision features from its image encoder.
The implementation uses the [openAI CLIP](https://github.com/openai/CLIP).
The extracted features are going to be of size `num_frames x 512`.
We additionally output timesteps in ms for each feature and fps of the video.

---

## Set up the Environment for CLIP
Setup `conda` environment. Requirements are in file `conda_env_torch_zoo.yml`
```bash
# it will create a new conda environment called 'torch_zoo' on your machine
conda env create -f conda_env_torch_zoo.yml
```

---

## Minimal Working Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13BamfEryOz0z_M-zWw5MHdf4lVvBtwj0?usp=sharing)

Activate the environment
```bash
conda activate torch_zoo
```

and extract features at 1 fps from `./sample/v_GGSY1Qvo990.mp4` video and output results.
```bash
python main.py \
    feature_type=clip \
    model_name=ViT-B/32 \
    extraction_fps=1 \
    video_paths="[./sample/v_GGSY1Qvo990.mp4]" \
    on_extraction=print
```

---

## Examples
Start by activating the environment
```bash
conda activate torch_zoo
```

It is pretty much the same procedure as with other features. 
Here we take `ViT/B-32` as an example, but we also support `ViT-B/16`, `RN50x16`, `RN50x4`, `RN101`, `RN50` and others in [openAI CLIP implementation](https://github.com/openai/CLIP).
In addition, if you want to use your weights, you need to copy your weights to `models/clip/checkpoints`, rename it `CLIP-custom.pth` and specify `model_name=custom`.
```bash
python main.py \
    feature_type=clip \
    model_name=ViT-B/32 \
    device_ids="[0, 2]" \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```
If you would like to save the features, use `--on_extraction save_numpy` (or `save_pickle`) – by default, the features are saved in `./output/` or where `--output_path` specifies. In the case of frame-wise features, besides features, it also saves timestamps in ms and the original fps of the video into the same folder with features.
```bash
python main.py \
    feature_type=clip \
    model_name=ViT-B/32 \
    device_ids="[0, 2]" \
    on_extraction=save_numpy \
    file_with_video_paths=./sample/sample_video_paths.txt
```
We may increase the extraction speed with batching. Therefore, frame-wise features have `--batch_size` argument, which defaults to `1`.
```bash
python main.py \
    feature_type=clip \
    model_name=ViT-B/32 \
    device_ids="[0, 2]" \
    batch_size=128 \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```
If you would like to extract features at a certain fps, add `--extraction_fps` argument
```bash
python main.py \
    feature_type=clip \
    model_name=ViT-B/32 \
    device_ids="[0, 2]" \
    extraction_fps=5 \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```

If you would like to verify the extracted features, you can
set `show_pred=true` and provide several sentences with `pred_texts` argument.
The value of `pred_texts` should be a list of strings. 
The probability that each frame corresponds to all the sentences you provide will be output.
```bash
python main.py \
    feature_type=clip \
    model_name=ViT-B/32 \
    device_ids="[0, 2]" \
    extraction_fps=1 \
    pred_texts="['a dog smiles', 'a man is lifting']" \
    show_pred=true \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```
You will get the output like:
```
Video #0: a dog smiles | a man is lifting
frame 0: [0.987789807340838, 0.012210192659162062]
frame 1: [0.9896199429752918, 0.010380057024708175]
frame 2: [0.9906664623528284, 0.009333537647171522]
frame 3: [0.9862671731489673, 0.013732826851032602]
frame 4: [0.9835191807313155, 0.01648081926868454]
...
Video #1: a dog smiles | a man is lifting
frame 0: [4.413239860117193e-06, 0.9999955867601399]
frame 1: [1.1625370081518226e-05, 0.9999883746299185]
frame 2: [1.7846816610242526e-06, 0.999998215318339]
frame 3: [1.029716861941584e-06, 0.999998970283138]
frame 4: [4.617640034529085e-06, 0.9999953823599654]
...
```

---

## Credits
1. The [openAI CLIP implementation](https://github.com/openai/CLIP).
2. The [CLIP paper](https://arxiv.org/abs/2103.00020)

---

## License
The openAI CLIP implementation code is under MIT.
