Just steps to install conda and create a new environment from scratch.
```bash
conda create -n video_features
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge omegaconf scipy tqdm pytest opencv
# +CLIP
conda install -c conda-forge ftfy regex
# vggish
conda install -c conda-forge resampy pysoundfile
# timm models
pip install timm
```
