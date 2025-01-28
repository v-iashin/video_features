# Setup Environment

## Using the YAML File

If you want to quickly set up the `conda` environment with all the required dependencies, use the `conda_env.yml` file. Run the following command:

```bash
# it will create a new conda environment called 'video_features' on your machine
conda env create -f conda_env.yml
```

## From Scratch

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
