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

RUN echo $'name: torch_zoo\n\
channels:\n\
  - pytorch\n\
  - conda-forge\n\
  - defaults\n\
dependencies:\n\
  - _libgcc_mutex=0.1=main\n\
  - antlr-python-runtime=4.9.3=pyhd8ed1ab_1\n\
  - attrs=22.1.0=pyh71513ae_1\n\
  - av=8.0.2=py38he20a9df_1\n\
  - backports=1.0=py_2\n\
  - backports.functools_lru_cache=1.6.4=pyhd8ed1ab_0\n\
  - blas=1.0=mkl\n\
  - bzip2=1.0.8=h516909a_3\n\
  - ca-certificates=2022.6.15=ha878542_0\n\
  - certifi=2022.6.15=py38h578d9bd_0\n\
  - cffi=1.14.6=py38h400218f_0\n\
  - cudatoolkit=11.0.221=h6bb024c_0\n\
  - ffmpeg=4.3.1=h167e202_0\n\
  - freetype=2.10.2=h5ab3b9f_0\n\
  - ftfy=6.1.1=pyhd8ed1ab_0\n\
  - gettext=0.19.8.1=h5e8e0c9_1\n\
  - gmp=6.2.0=he1b5a44_2\n\
  - gnutls=3.6.13=h79a8f9a_0\n\
  - iniconfig=1.1.1=pyh9f0ad1d_0\n\
  - intel-openmp=2020.2=254\n\
  - jpeg=9b=h024ee3a_2\n\
  - lame=3.100=h14c3975_1001\n\
  - lcms2=2.11=h396b838_0\n\
  - ld_impl_linux-64=2.33.1=h53a641e_7\n\
  - libedit=3.1.20191231=h14c3975_1\n\
  - libffi=3.3=he6710b0_2\n\
  - libflac=1.3.3=he1b5a44_0\n\
  - libgcc-ng=9.1.0=hdf63c60_0\n\
  - libgfortran-ng=7.3.0=hdf63c60_0\n\
  - libiconv=1.16=h516909a_0\n\
  - libllvm10=10.0.1=he513fc3_3\n\
  - libogg=1.3.2=h516909a_1002\n\
  - libpng=1.6.37=hbc83047_0\n\
  - libsndfile=1.0.29=he1b5a44_0\n\
  - libstdcxx-ng=9.1.0=hdf63c60_0\n\
  - libtiff=4.1.0=h2733197_1\n\
  - libuv=1.40.0=h7b6447c_0\n\
  - libvorbis=1.3.7=he1b5a44_0\n\
  - llvmlite=0.36.0=py38h612dafd_4\n\
  - lz4-c=1.9.2=he6710b0_1\n\
  - mkl=2020.2=256\n\
  - mkl-service=2.3.0=py38he904b0f_0\n\
  - mkl_fft=1.2.0=py38h23d657b_0\n\
  - mkl_random=1.1.1=py38h0573a6f_0\n\
  - ncurses=6.2=he6710b0_1\n\
  - nettle=3.4.1=h1bed415_1002\n\
  - ninja=1.10.1=py38hfd86e86_0\n\
  - numba=0.53.1=py38ha9443f7_0\n\
  - numpy=1.19.1=py38hbc911f0_0\n\
  - numpy-base=1.19.1=py38hfa32c7d_0\n\
  - olefile=0.46=py_0\n\
  - omegaconf=2.1.1=py38h578d9bd_1\n\
  - openh264=2.1.1=h8b12597_0\n\
  - openssl=1.1.1q=h7f8727e_0\n\
  - packaging=21.3=pyhd8ed1ab_0\n\
  - pillow=7.2.0=py38hb39fc2d_0\n\
  - pip=20.2.2=py38_0\n\
  - pluggy=1.0.0=py38h578d9bd_3\n\
  - py=1.11.0=pyh6c4a22f_0\n\
  - pycparser=2.21=pyhd8ed1ab_0\n\
  - pyparsing=3.0.9=pyhd8ed1ab_0\n\
  - pysoundfile=0.10.3.post1=pyhd3deb0d_0\n\
  - pytest=7.1.2=py38h578d9bd_0\n\
  - python=3.8.5=h7579374_1\n\
  - python_abi=3.8=1_cp38\n\
  - pytorch=1.7.1=py3.8_cuda11.0.221_cudnn8.0.5_0\n\
  - pyyaml=5.3.1=py38h8df0ef7_1\n\
  - readline=8.0=h7b6447c_0\n\
  - regex=2022.3.15=py38h7f8727e_0\n\
  - resampy=0.2.2=py_0\n\
  - scipy=1.5.2=py38h0b6359f_0\n\
  - setuptools=49.6.0=py38_0\n\
  - six=1.15.0=py_0\n\
  - sqlite=3.33.0=h62c20be_0\n\
  - tbb=2020.2=hc9558a2_0\n\
  - tk=8.6.10=hbc83047_0\n\
  - tomli=2.0.1=pyhd8ed1ab_0\n\
  - torchaudio=0.7.2=py38\n\
  - torchvision=0.8.2=py38_cu110\n\
  - tqdm=4.49.0=py_0\n\
  - typing_extensions=4.0.1=pyha770c72_0\n\
  - wcwidth=0.2.5=pyh9f0ad1d_2\n\
  - wheel=0.35.1=py_0\n\
  - x264=1!152.20180806=h14c3975_0\n\
  - xz=5.2.5=h7b6447c_0\n\
  - yaml=0.2.5=h516909a_0\n\
  - zlib=1.2.11=h7b6447c_3\n\
  - zstd=1.4.5=h9ceee32_0\n\
  - pip:\n\
    - opencv-python==4.4.0.44\n\
' >> conda_env_torch_zoo.yml

RUN echo $'name: pwc\n\
channels:\n\
  - pytorch\n\
  - conda-forge\n\
  - defaults\n\
dependencies:\n\
  - _libgcc_mutex=0.1=main\n\
  - antlr-python-runtime=4.9.3=pyhd8ed1ab_1\n\
  - attrs=22.1.0=pyh71513ae_1\n\
  - autopep8=1.4.4=py_0\n\
  - av=7.0.1=py37hfbe2fac_1\n\
  - backports=1.0=py_2\n\
  - backports.functools_lru_cache=1.6.4=pyhd8ed1ab_0\n\
  - blas=1.0=mkl\n\
  - bzip2=1.0.8=h7b6447c_0\n\
  - ca-certificates=2022.6.15=ha878542_0\n\
  - certifi=2022.6.15=py37h89c1867_0\n\
  - cffi=1.14.0=py37he30daa8_1\n\
  - cudatoolkit=10.0.130=0\n\
  - cudnn=7.3.1=cuda10.0_0\n\
  - cupy=5.1.0=py37hc0ce245_0\n\
  - fastrlock=0.4=py37he6710b0_0\n\
  - ffmpeg=4.1.3=h167e202_0\n\
  - flake8=3.8.2=py_0\n\
  - freetype=2.9.1=h8a8886c_1\n\
  - ftfy=6.1.1=pyhd8ed1ab_0\n\
  - gmp=6.1.2=h6c8ec71_1\n\
  - gnutls=3.6.5=h71b1129_1002\n\
  - h5py=2.8.0=py37h989c5e5_3\n\
  - hdf5=1.10.2=hba1933b_1\n\
  - importlib-metadata=1.6.0=py37_0\n\
  - importlib_metadata=1.5.0=py37_0\n\
  - iniconfig=1.1.1=pyh9f0ad1d_0\n\
  - intel-openmp=2020.1=217\n\
  - jpeg=9b=h024ee3a_2\n\
  - lame=3.100=h7b6447c_0\n\
  - ld_impl_linux-64=2.33.1=h53a641e_7\n\
  - libedit=3.1.20181209=hc058e9b_0\n\
  - libffi=3.3=he6710b0_1\n\
  - libgcc-ng=9.1.0=hdf63c60_0\n\
  - libgfortran-ng=7.3.0=hdf63c60_0\n\
  - libiconv=1.15=h63c8f33_5\n\
  - libpng=1.6.37=hbc83047_0\n\
  - libstdcxx-ng=9.1.0=hdf63c60_0\n\
  - libtiff=4.1.0=h2733197_0\n\
  - mccabe=0.6.1=py37_1\n\
  - mkl=2020.1=217\n\
  - mkl-service=2.3.0=py37he904b0f_0\n\
  - mkl_fft=1.0.15=py37ha843d7b_0\n\
  - mkl_random=1.1.1=py37h0573a6f_0\n\
  - nccl=1.3.5=cuda10.0_0\n\
  - ncurses=6.2=he6710b0_1\n\
  - nettle=3.4.1=hbb512f6_0\n\
  - ninja=1.9.0=py37hfd86e86_0\n\
  - numpy=1.15.4=py37h7e9f1db_0\n\
  - numpy-base=1.15.4=py37hde5b4d6_0\n\
  - olefile=0.46=py37_0\n\
  - omegaconf=2.1.1=py37h89c1867_1\n\
  - openh264=1.8.0=hd408876_0\n\
  - openssl=1.1.1q=h7f8727e_0\n\
  - packaging=21.3=pyhd8ed1ab_0\n\
  - pillow=6.1.0=py37h34e0f95_0\n\
  - pip=20.0.2=py37_3\n\
  - pluggy=1.0.0=py37h89c1867_3\n\
  - py=1.11.0=pyh6c4a22f_0\n\
  - pycodestyle=2.6.0=py_0\n\
  - pycparser=2.20=py_0\n\
  - pyflakes=2.2.0=py_0\n\
  - pyparsing=3.0.9=pyhd8ed1ab_0\n\
  - pytest=7.1.2=py37h89c1867_0\n\
  - python=3.7.7=hcff3b4d_5\n\
  - python_abi=3.7=1_cp37m\n\
  - pytorch=1.2.0=py3.7_cuda10.0.130_cudnn7.6.2_0\n\
  - pyyaml=5.3.1=py37hb5d75c8_1\n\
  - readline=8.0=h7b6447c_0\n\
  - regex=2022.3.15=py37h7f8727e_0\n\
  - rope=0.17.0=py_0\n\
  - scipy=1.5.2=py37h0b6359f_0\n\
  - setuptools=46.4.0=py37_0\n\
  - six=1.14.0=py37_0\n\
  - sqlite=3.31.1=h62c20be_1\n\
  - tk=8.6.8=hbc83047_0\n\
  - tomli=2.0.1=pyhd8ed1ab_0\n\
  - torchvision=0.4.0=py37_cu100\n\
  - tqdm=4.46.0=py_0\n\
  - typing_extensions=4.0.1=pyha770c72_0\n\
  - wcwidth=0.2.5=pyh9f0ad1d_2\n\
  - wheel=0.34.2=py37_0\n\
  - x264=1!152.20180806=h7b6447c_0\n\
  - xz=5.2.5=h7b6447c_0\n\
  - yaml=0.2.5=h516909a_0\n\
  - zipp=3.1.0=py_0\n\
  - zlib=1.2.11=h7b6447c_3\n\
  - zstd=1.3.7=h0b5b093_0\n\
  - pip:\n\
    - opencv-python==4.4.0.44\n\
' >> conda_env_pwc.yml

RUN conda env create -f conda_env_torch_zoo.yml
RUN conda env create -f conda_env_pwc.yml
RUN conda clean -afy
RUN rm ./Miniconda3-latest-Linux-x86_64.sh

RUN sudo apt-get -qq install libglib2.0-0 libsndfile1 libsm6 libxext6 libxrender-dev libgl1

SHELL ["conda", "run", "-n", "torch_zoo", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "torch_zoo"]
