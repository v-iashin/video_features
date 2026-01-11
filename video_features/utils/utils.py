import argparse
import os
import pickle
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Union
import platform

import numpy as np
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig

# Use package-relative paths that work both in development and when installed
def _get_package_path(relative_path: str) -> str:
    """Get absolute path to a file in the package."""
    utils_dir = Path(__file__).parent
    # Try to find the file relative to utils directory (when installed)
    installed_path = utils_dir / relative_path
    if installed_path.exists():
        return str(installed_path)
    # Fallback: try relative to project root (development mode)
    video_features_dir = utils_dir.parent  # video_features
    package_root = video_features_dir.parent  # project root
    return str(package_root / relative_path)

IMAGENET1K_CLASS_PATH = _get_package_path('IN1K_label_map.txt')
IMAGENET21K_CLASS_PATH = _get_package_path('IN21K_label_map.txt')
KINETICS_CLASS_PATH = _get_package_path('K400_label_map.txt')


def show_predictions_on_dataset(logits: torch.FloatTensor, dataset: Union[str, List]):
    '''Prints out predictions for each feature

    Args:
        logits (torch.FloatTensor): after-classification layer vector (B, classes)
        dataset (str): which dataset to use to show the predictions on. In ('imagenet', 'kinetics')
    '''
    if dataset == 'kinetics':
        dataset_classes = [x.strip() for x in open(KINETICS_CLASS_PATH)]
    elif dataset == 'imagenet1k':
        dataset_classes = [x.strip() for x in open(IMAGENET1K_CLASS_PATH)]
    elif dataset == 'imagenet21k':
        dataset_classes = [x.strip() for x in open(IMAGENET21K_CLASS_PATH)]
    elif isinstance(dataset, list):
        dataset_classes = dataset
    else:
        raise NotImplementedError

    # Show predictions
    softmaxes = F.softmax(logits, dim=-1)
    top_val, top_idx = torch.sort(softmaxes, dim=-1, descending=True)

    k = 5
    logits_score = logits.gather(1, top_idx[:, :k]).tolist()
    softmax_score = softmaxes.gather(1, top_idx[:, :k]).tolist()
    class_labels = [[dataset_classes[idx] for idx in i_row] for i_row in top_idx[:, :k]]

    for b in range(len(logits)):
        # header
        print('  Logits | Prob. | Label ')
        for (logit, smax, cls) in zip(logits_score[b], softmax_score[b], class_labels[b]):
            # rows
            print(f'{logit:8.3f} | {smax:.3f} | {cls}')
        print()

def make_path(output_root, video_path, output_key, ext):
    # extract file name and change the extention
    fname = f'{Path(video_path).stem}_{output_key}{ext}'
    # construct the paths to save the features
    return os.path.join(output_root, fname)

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
    '''Checks user arguments.

    Args:
        args (Union[argparse.Namespace, DictConfig]): Parsed user arguments
    '''
    if 'device_ids' in args:
        print('WARNING:')
        print('Running feature extraction on multiple devices in a _single_ process is no longer supported.')
        print('To use several GPUs, you simply need to start the extraction with another GPU ordinal.')
        print('For instance, in one terminal: `device="cuda:0"` and `device="cuda:1"` in the second, etc.')
        print(f'Your device specification (device_ids={args.device_ids}) is converted to `device="cuda:0"`.')
        args.device = 'cuda:0'
    if 'cuda' in args.device and not torch.cuda.is_available():
        print(f'A GPU was attempted to use but the system does not have one. Going to use CPU...')
        args.device = 'cpu'
    assert args.file_with_video_paths or args.video_paths, '`video_paths` or `file_with_video_paths` must be specified'
    filenames = [Path(p).stem for p in form_list_from_user_input(args.video_paths, args.file_with_video_paths)]
    assert len(filenames) == len(set(filenames)), 'Non-unique filenames. See video_features/issues/54'
    # Use Path.resolve() instead of os.path.relpath() to handle cross-drive paths on Windows
    output_path_abs = Path(args.output_path).resolve()
    tmp_path_abs = Path(args.tmp_path).resolve()
    assert output_path_abs != tmp_path_abs, 'The same path for out & tmp'
    if args.show_pred:
        if args.feature_type == 'vggish':
            print('Showing class predictions is not implemented for VGGish')
    # if args.feature_type == 'r21d':
    #     message = 'torchvision.read_video only supports extraction at orig fps. Remove this argument.'
    #     assert args.extraction_fps is None, message
    if args.feature_type == 'i3d':
        message = f'I3D model does not support inputs shorter than 10 timestamps. You have: {args.stack_size}'
        if args.stack_size is not None:
            assert args.stack_size >= 10, message
    if args.feature_type == 'pwc' or (args.feature_type == 'i3d' and args.flow_type == 'pwc'):
        raise NotImplementedError(
            'PWC is no longer supported. Please see: https://github.com/v-iashin/video_features/pull/112')
    if args.feature_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'r21d']:
        if args.keep_tmp_files:
            print('If you want to keep frames while extracting features, please create an issue')
    if args.feature_type == 'timm':
        assert args.model_name is not None, 'Please specify `model_name` for timm models; '\
                                            'eg `efficientnet_b0`'
    if 'batch_size' in args:
        assert args.batch_size is not None, f'Please specify `batch_size`. It is {args.batch_size} now'
    if 'extraction_fps' in args and 'extraction_total' in args:
        assert not (args.extraction_fps is not None and args.extraction_total is not None),\
            '`fps` and `total` is mutually exclusive'

    # patch_output_paths
    # preprocess paths
    subs = [args.feature_type]
    if hasattr(args, 'model_name'):
        subs.append(args.model_name)
        # may add `finetuned_on` item
    real_output_path = args.output_path
    real_tmp_path = args.tmp_path
    for p in subs:
        # some model use `/` e.g. ViT-B/16
        real_output_path = os.path.join(real_output_path, p.replace("/", "_"))
        real_tmp_path = os.path.join(real_tmp_path, p.replace("/", "_"))
    args.output_path = real_output_path
    args.tmp_path = real_tmp_path


def form_list_from_user_input(
        video_paths: Union[str, ListConfig, None] = None,
        file_with_video_paths: str = None,
        to_shuffle: bool = True,
    ) -> list:
    '''User specifies either list of videos in the cmd or a path to a file with video paths. This function
       transforms the user input into a list of paths. Files are expected to be formatted with a single
       video-path in each line.

    Args:
        video_paths (Union[str, ListConfig, None], optional): a list of video paths. Defaults to None.
        file_with_video_paths (str, optional): a path to a file with video files for extraction.
                                               Defaults to None.
        to_shuffle (bool, optional): if the list of paths should be shuffled. If True is should prevent
                                     potential worker collisions (two workers process the same video)

    Returns:
        list: list with paths
    '''
    if file_with_video_paths is None:
        path_list = [video_paths] if isinstance(video_paths, str) else list(video_paths)
        # TODO: the following `if` could be redundant
        # ListConfig does not support indexing with tensor scalars, e.g. tensor(1, device='cuda:0')
        if isinstance(video_paths, ListConfig):
            path_list = list(path_list)
    else:
        with open(file_with_video_paths) as rfile:
            # remove carriage return
            path_list = [line.replace('\n', '') for line in rfile.readlines()]
            # remove empty lines
            path_list = [path for path in path_list if len(path) > 0]

    # sanity check: prints paths which do not exist
    for path in path_list:
        if not Path(path).exists():
            print(f'The path does not exist: {path}')

    if to_shuffle:
        random.shuffle(path_list)

    return path_list


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    # Determine the platform on which the program is running
    if platform.system().lower() == 'windows':
        result = subprocess.run(['where', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ffmpeg_path = result.stdout.decode('utf-8').replace('\r\n', '')
    else:
        result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path


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


def get_package_file_path(relative_path: str) -> Path:
    """Get absolute path to a file within the video_features package.
    
    This function works both when the package is installed and in development mode.
    It tries multiple methods to locate files within the package.
    
    Args:
        relative_path: Path relative to video_features package root
                      (e.g., 'models/raft/checkpoints/raft-sintel.pth')
    
    Returns:
        Path object to the file
    """
    # Try to use importlib.resources first (for installed packages)
    try:
        import importlib.resources as pkg_resources
        # Try files() API first (Python 3.9+)
        try:
            package_files = pkg_resources.files('video_features')
            file_path = package_files / relative_path
            if file_path.exists():
                return Path(str(file_path))
        except (AttributeError, TypeError):
            # Fallback to path() API (Python 3.7-3.8)
            try:
                with pkg_resources.path('video_features', relative_path) as file_path:
                    return Path(str(file_path))
            except FileNotFoundError:
                pass
    except (ImportError, ModuleNotFoundError):
        # Fallback to pkg_resources (setuptools)
        try:
            import pkg_resources
            file_path = Path(pkg_resources.resource_filename('video_features', relative_path))
            if file_path.exists():
                return file_path
        except:
            pass
    
    # Fallback: look relative to package directory (development mode)
    utils_dir = Path(__file__).parent
    video_features_dir = utils_dir.parent  # video_features
    file_path = video_features_dir / relative_path
    if file_path.exists():
        return file_path
    
    # Last resort: look at project root (for backward compatibility)
    package_root = video_features_dir.parent  # project root
    file_path = package_root / relative_path
    return file_path


def build_cfg_path(feature_type: str) -> os.PathLike:
    '''Makes a path to the default config file for each feature family.

    Args:
        feature_type (str): the type (e.g. 'vggish')

    Returns:
        os.PathLike: the path to the default config for the type
    '''
    # Try to use importlib.resources first (for installed packages)
    # Python 3.9+ uses files() API, older versions use path()
    try:
        import importlib.resources as pkg_resources
        # Try files() API first (Python 3.9+)
        try:
            config_files = pkg_resources.files('video_features.configs')
            config_path = config_files / f'{feature_type}.yml'
            if config_path.exists():
                return str(config_path)
        except (AttributeError, TypeError):
            # Fallback to path() API (Python 3.7-3.8)
            with pkg_resources.path('video_features.configs', f'{feature_type}.yml') as config_path:
                return str(config_path)
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        # Fallback to pkg_resources (setuptools)
        try:
            import pkg_resources
            config_path = Path(pkg_resources.resource_filename('video_features', f'configs/{feature_type}.yml'))
            if config_path.exists():
                return str(config_path)
        except:
            pass
    
    # Fallback: look relative to package directory (development mode)
    utils_dir = Path(__file__).parent
    video_features_dir = utils_dir.parent  # video_features
    config_path = video_features_dir / 'configs' / f'{feature_type}.yml'
    if config_path.exists():
        return config_path
    
    # Last resort: look at project root (for backward compatibility)
    package_root = video_features_dir.parent  # project root
    path = package_root / 'configs' / f'{feature_type}.yml'
    return path


def dp_state_to_normal(state_dict):
    '''Converts a torch.DataParallel checkpoint to regular'''
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module'):
            new_state_dict[k.replace('module.', '')] = v
    return new_state_dict


def load_numpy(fpath):
    return np.load(fpath)

def write_numpy(fpath, value):
    return np.save(fpath, value)

def load_pickle(fpath):
    return pickle.load(open(fpath, 'rb'))

def write_pickle(fpath, value):
    return pickle.dump(value, open(fpath, 'wb'))
