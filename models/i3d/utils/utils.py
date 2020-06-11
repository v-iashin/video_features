import os
import numpy as np
import cv2
import subprocess

from utils.utils import which_ffmpeg

from typing import Union

def center_crop(tensor, crop_size: int = 224):
    '''
        Takes a tensor of input (..., H, W) and performs a 
        center crop of provided size.
    '''
    H, W = tensor.size(-2), tensor.size(-1)

    from_H = ((H - crop_size) // 2)
    to_H = from_H + crop_size

    from_W = ((W - crop_size) // 2)
    to_W = from_W + crop_size

    return tensor[..., from_H:to_H, from_W:to_W]


def form_iter_list(framepaths, step_size, stack_size, phase=None, min_ratio_for_cycling=0.2):
    '''
        Forms a list of lists each containing paths for a rgb stack.
        The window of a specified stack_size applied for the given list
        with step_size. Since most of the time framepaths cannot be fully
        devided into stacks, the last list is cycled if the number of 
        remaining paths in it is higher than a specified threshold.
    '''
    framelist_empty = len(framepaths) == 0

    if framelist_empty:
        return []

    # calc how many full stacks can be formed out of framepaths
    full_stack_num = (len(framepaths) - stack_size) // step_size + 1

    stacks = []

    # how many elements are in the last list
    reminder = len(framepaths) % stack_size

    # if the segment is too short but is large enough to cycle
    # we want to repeat the frames regardless of the number
    # of frames on validation (only but Todo: for train as well)
    # changing this for validation (and train but requires recalculation of meta and i3d)
    frame_num_is_lower_than_stacksize = len(framepaths) < stack_size
    reminder_is_large_enough = reminder >= min_ratio_for_cycling * stack_size

    # if frame_num_is_lower_than_stacksize and (reminder_is_large_enough or not train):
    if frame_num_is_lower_than_stacksize and (reminder_is_large_enough or (phase != 'train')):
        remained_paths = np.array(framepaths)
        # performs cycling
        cycled_remained_paths = np.resize(remained_paths, stack_size)
        stacks.append(list(cycled_remained_paths))

        return stacks

    # else form full stack and deal with the remaining in the same fashion.
    for i in range(full_stack_num):
        start_idx = i * step_size
        end_idx = start_idx + stack_size
        stacks.append(framepaths[start_idx:end_idx])

    # if the number of remaining elements in the list
    # is high enough, cycle those elements until stack_size is reached
    if reminder_is_large_enough:
        start_idx = full_stack_num * step_size
        remained_paths = np.array(framepaths[start_idx:])
        # performs cycling
        cycled_remained_paths = np.resize(remained_paths, stack_size)
        stacks.append(list(cycled_remained_paths))

    return stacks

def extract_frames_from_video(video_path: str, extraction_fps: Union[str, None], 
                              min_side_target_size: int, tmp_path: str) -> str:
    '''Extracts frames from a video using the specified path.

    Arguments:
        video_path {str} -- path to a video
        extraction_fps {Union[str, None]} -- fps for extraction
        min_side_target_size {int} -- min(height, width) of the resized video
        tmp_path {str} -- path where to extract frames

    Returns:
        [str] -- path to the folder with extracted frames
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'

    video = cv2.VideoCapture(video_path)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)

    if extraction_fps is None:
        original_fps = video.get(cv2.CAP_PROP_FPS)
        extraction_fps = original_fps

    # Make dir for frames
    folder_name = os.path.split(video_path)[-1].replace('.mp4', '_frames')
    frames_path = os.path.join(tmp_path, folder_name)
    if os.path.exists(frames_path):
        print(f'Warning: folder {frames_path} already exists. Possibly messed up with frames')
    else:
        os.makedirs(frames_path, exist_ok=True)
    
    # vertical/horizontal video handling
    if width > height:
        size = f'-1:{min_side_target_size}'
    else:
        size = f'{min_side_target_size}:-1'
    
    # Extract frames: call ffmpeg
    extract_cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic -y -i {video_path}'
    extract_cmd += f' -vf scale={size} -r {extraction_fps} -q:v 2 {frames_path}/%6d.jpg'

    subprocess.call(extract_cmd.split())

    return frames_path
