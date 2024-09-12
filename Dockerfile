FROM ubuntu:18.04

# RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN apt-get install -y sudo

RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu

SHELL ["/bin/bash", "-c"]

RUN sudo apt-get -qq install curl vim git zip

WORKDIR /home/ubuntu/

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/home/ubuntu/miniconda3/bin:$PATH"
RUN echo ". /home/ubuntu/miniconda3/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash
RUN conda config --set auto_activate_base false

COPY conda_env.yml .

RUN conda env create -f conda_env.yml
RUN conda clean -afy
RUN rm ./Miniconda3-latest-Linux-x86_64.sh

RUN sudo apt-get -qq install libglib2.0-0 libsndfile1 libsm6 libxext6 libxrender-dev libgl1

SHELL ["conda", "run", "-n", "video_features", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "video_features"]
