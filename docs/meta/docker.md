# Docker support
For your convenience, there is also a [Docker image](https://hub.docker.com/r/iashin/video_features)
with the pre-installed environments that supports all models.
The Docker image does not have the `video_features` library inside which allows you
to tweak the code locally, mount the new version, and just use the container as an environment.
It is assumed that you have
[Docker](https://docs.docker.com/desktop/#download-and-install)
and
[nvidia-container-runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu)
installed.

## Setup

Start by cloning the repo locally if you haven't done it already
```bash
git clone https://github.com/v-iashin/video_features.git
```

Download the docker image or build it yourself:
```bash
docker pull iashin/video_features
# preventing newer versions of the image to be downloaded unexpectedly
docker tag iashin/video_features video_features
# or
# docker build - < ./video_features/Dockerfile --tag video_features
```

Once it is done, mount (`--mount`) the cloned repository folder,
and initialize a container with the 0th GPU but remember:
**just like with any mount, a change from inside of the**
**container will be reflected in the mounted folder (`/absolute/path/to/video_features/`).**:
```bash
docker run -it \
    --mount type=bind,source="/absolute/path/to/video_features/",destination="/home/ubuntu/video_features/" \
    --shm-size 8G \
    -it --gpus '"device=0"' \
    video_features:latest \
    bash
# and you should get the bash shell:
# ubuntu@56b1bf77a20c:~$
```

Check if a GPU is available to PyTorch:
```bash
# ubuntu@56b1bf77a20c:~$
python -c "import torch; print(torch.cuda.is_available())"
# True
```

Finally, try to extract video features:
```bash
# cd to `./video_features`
# ubuntu@56b1bf77a20c:~/video_features $
python main.py \
    feature_type=r21d \
    device="cuda:0" \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"
```

## Extract features from custom videos

You need to mount the folders with video files before you start the container.

If the folder with custom videos is already in `./video_features`, you
don't have to do anything.
**Any changes from inside of the container will be reflected in your original dataset (use a backup!).**
Here is an example of mounting a folder from somewhere else
(mounts `/absolute/path/somewhere/else/` to
`/home/ubuntu/video_features/dataset`):
```bash
docker run -it \
    --mount type=bind,source="/absolute/path/to/video_features/",destination="/home/ubuntu/video_features/" \
    --mount type=bind,source="/absolute/path/somewhere/else/",destination="/home/ubuntu/video_features/dataset/" \
    --shm-size 8G \
    -it --gpus '"device=0"' \
    video_features:latest \
    bash
# ubuntu@56b1bf77a20c:~$
ls ./video_features
# ... dataset ...
```

If you want to save outputs to another folder on your local machine,
you may want to mount it as well: e.g. by adding
```bash
...
    --mount type=bind,source="/absolute/path/somewhere/else/",destination="/home/ubuntu/video_features/output/" \
...
```

Then, run your command.
For instance:
```bash
# cd to `./video_features`
# ubuntu@56b1bf77a20c:~/video_features $
python main.py \
    feature_type=r21d \
    device="cuda:0" \
    video_paths="[./dataset/vid_1.mp4, ./dataset/vid_2.mp4]" \
    on_extraction="save_numpy"
# you should have features in `./output`
# (and in the source location if you mount to it)
```


## Switching conda environments
By default, the `torch_zoo` environment is activated once you attach the shell.
The image supports both conda environments and you can switch it simply as follows:
```bash
# ubuntu@56b1bf77a20c:~$
conda activate pwc
conda deactivate
conda activate torch_zoo
# which python
```
