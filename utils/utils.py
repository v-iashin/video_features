import argparse
import os
import pickle
import subprocess
from pathlib import Path
from typing import Dict, Union

import numpy as np
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig

IMAGENET_CLASS_PATH = './utils/IN_label_map.txt'
KINETICS_CLASS_PATH = './utils/K400_label_map.txt'


def show_predictions_on_dataset(logits: torch.FloatTensor, dataset: str):
    '''Prints out predictions for each feature

    Args:
        logits (torch.FloatTensor): after-classification layer vector (B, classes)
        dataset (str): which dataset to use to show the predictions on. In ('imagenet', 'kinetics')
    '''
    if dataset == 'kinetics':
        path_to_class_list = KINETICS_CLASS_PATH
    elif dataset == 'imagenet':
        path_to_class_list = IMAGENET_CLASS_PATH
    else:
        raise NotImplementedError

    dataset_classes = [x.strip() for x in open(path_to_class_list)]

    # Show predictions
    softmaxes = F.softmax(logits, dim=-1)
    top_val, top_idx = torch.sort(softmaxes, dim=-1, descending=True)

    k = 5
    logits_score = logits.gather(1, top_idx[:, :k]).tolist()
    softmax_score = softmaxes.gather(1, top_idx[:, :k]).tolist()
    class_labels = [[dataset_classes[idx] for idx in i_row] for i_row in top_idx[:, :k]]

    for b in range(len(logits)):
        for (logit, smax, cls) in zip(logits_score[b], softmax_score[b], class_labels[b]):
            print(f'{logit:.3f} {smax:.3f} {cls}')
        print()

def action_on_extraction(feats_dict: Dict[str, np.ndarray], video_path, output_path, on_extraction: str):
    '''What is going to be done with the extracted features.

    Args:
        feats_dict (Dict[str, np.ndarray]): A dict with features and possibly some meta. Key will be used as
                                            suffixes to the saved files if `save_numpy` or `save_pickle` is
                                            used.
        video_path (str): A path to the video.
        on_extraction (str): What to do with the features on extraction.
        output_path (str): Where to save the features if `save_numpy` or `save_pickle` is used.
    '''
    # since the features are enclosed in a dict with another meta information we will iterate on kv
    for key, value in feats_dict.items():
        if on_extraction == 'print':
            print(key)
            print(value)
            print(f'max: {value.max():.8f}; mean: {value.mean():.8f}; min: {value.min():.8f}')
            print()
        elif on_extraction == 'save_numpy':
            # make dir if doesn't exist
            os.makedirs(output_path, exist_ok=True)
            # extract file name and change the extention
            fname = f'{Path(video_path).stem}_{key}.npy'
            # construct the paths to save the features
            fpath = os.path.join(output_path, fname)
            if key != 'fps' and len(value) == 0:
                print(f'Warning: the value is empty for {key} @ {fpath}')
            # save the info behind the each key
            np.save(fpath, value)
        elif on_extraction == 'save_pickle':
            # make dir if doesn't exist
            os.makedirs(output_path, exist_ok=True)
            # extract file name and change the extention
            fname = f'{Path(video_path).stem}_{key}.pkl'
            # construct the paths to save the features
            fpath = os.path.join(output_path, fname)
            if key != 'fps' and len(value) == 0:
                print(f'Warning: the value is empty for {key} @ {fpath}')
            # save the info behind the each key
            pickle.dump(value, open(fpath, 'wb'))
        else:
            raise NotImplementedError(f'on_extraction: {on_extraction} is not implemented')

def form_slices(size: int, stack_size: int, step_size: int) -> list((int, int)):
    '''print(form_slices(100, 15, 15) - example'''
    slices = []
    # calc how many full stacks can be formed out of framepaths
    full_stack_num = (size - stack_size) // step_size + 1
    for i in range(full_stack_num):
        start_idx = i * step_size
        end_idx = start_idx + stack_size
        slices.append((start_idx, end_idx))
    return slices


def sanity_check(args: Union[argparse.Namespace, DictConfig]):
    '''Checks the prased user arguments.

    Args:
        args (Union[argparse.Namespace, DictConfig]): Parsed user arguments
    '''
    assert args.file_with_video_paths or args.video_paths, '`video_paths` or `file_with_video_paths` must be specified'
    assert os.path.relpath(args.output_path) != os.path.relpath(args.tmp_path), 'The same path for out & tmp'
    if args.show_pred:
        print('You want to see predictions. So, I will use only the first GPU from the list you specified.')
        args.device_ids = [args.device_ids[0]]
        if args.feature_type == 'vggish':
            print('Showing class predictions is not implemented for VGGish')
    # if args.feature_type == 'r21d':
    #     message = 'torchvision.read_video only supports extraction at orig fps. Remove this argument.'
    #     assert args.extraction_fps is None, message
    if args.feature_type == 'i3d':
        message = f'I3D model does not support inputs shorter than 10 timestamps. You have: {args.stack_size}'
        if args.stack_size is not None:
            assert args.stack_size >= 10, message
    if args.feature_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'r21d']:
        if args.keep_tmp_files:
            print('If you want to keep frames while extracting features, please create an issue')


def form_list_from_user_input(args: Union[argparse.Namespace, DictConfig]) -> list:
    '''User specifies either list of videos in the cmd or a path to a file with video paths. This function
    transforms the user input into a list of paths.

    Args:
        args (Union[argparse.Namespace, DictConfig]): Parsed user arguments

    Returns:
        list: list with paths
    '''
    if args.file_with_video_paths is None:
        path_list = args.video_paths
        # ListConfig does not support indexing with tensor scalars, e.g. tensor(1, device='cuda:0')
        if isinstance(args.video_paths, ListConfig):
            path_list = list(path_list)
    else:
        with open(args.file_with_video_paths) as rfile:
            # remove carriage return
            path_list = [line.replace('\n', '') for line in rfile.readlines()]
            # remove empty lines
            path_list = [path for path in path_list if len(path) > 0]

    # sanity check: prints paths which do not exist
    for path in path_list:
        not_exist = not os.path.exists(path)
        if not_exist:
            print(f'The path does not exist: {path}')

    return path_list


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path


def reencode_video_with_diff_fps(video_path: str, tmp_path: str, extraction_fps: float) -> str:
    '''Reencodes the video given the path and saves it to the tmp_path folder.

    Args:
        video_path (str): original video
        tmp_path (str): the folder where tmp files are stored (will be appended with a proper filename).
        extraction_fps (float): target fps value

    Returns:
        str: The path where the tmp file is stored. To be used to load the video from
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # form the path to tmp directory
    new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps.mp4')
    cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
    cmd += f'-y -i {video_path} -filter:v fps=fps={extraction_fps} {new_path}'
    subprocess.call(cmd.split())

    return new_path


def extract_wav_from_mp4(video_path: str, tmp_path: str) -> str:
    '''Extracts .wav file from .aac which is extracted from .mp4
    We cannot convert .mp4 to .wav directly. For this we do it in two stages: .mp4 -> .aac -> .wav

    Args:
        video_path (str): Path to a video
        audio_path_wo_ext (str):

    Returns:
        [str, str] -- path to the .wav and .aac audio
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # extract video filename from the video_path
    video_filename = os.path.split(video_path)[-1].replace('.mp4', '')

    # the temp files will be saved in `tmp_path` with the same name
    audio_aac_path = os.path.join(tmp_path, f'{video_filename}.aac')
    audio_wav_path = os.path.join(tmp_path, f'{video_filename}.wav')

    # constructing shell commands and calling them
    mp4_to_acc = f'{which_ffmpeg()} -hide_banner -loglevel panic -y -i {video_path} -acodec copy {audio_aac_path}'
    aac_to_wav = f'{which_ffmpeg()} -hide_banner -loglevel panic -y -i {audio_aac_path} {audio_wav_path}'
    subprocess.call(mp4_to_acc.split())
    subprocess.call(aac_to_wav.split())

    return audio_wav_path, audio_aac_path


def build_cfg_path(feature_type: str) -> os.PathLike:
    '''Makes a path to the default config file for each feature family.

    Args:
        feature_type (str): the type (e.g. 'vggish')

    Returns:
        os.PathLike: the path to the default config for the type
    '''
    path_base = Path('./configs')
    if feature_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        path = path_base / 'resnet.yml'
    else:
        path = path_base / f'{feature_type}.yml'
    return path
