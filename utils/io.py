import cv2
import numpy as np
from torch import Tensor
from pathlib import Path

from typing import Union, Optional, Callable, List, Tuple
from os import PathLike
import subprocess
import os

from .utils import which_ffmpeg


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


class VideoLoader:
    def __init__(self,
                 path: Union[str, PathLike],
                 batch_size: int = 1,
                 fps: Optional[int] = None,
                 total: Optional[int] = None,
                 tmp_path: Optional[Union[str, PathLike]] = 'tmp',
                 keep_tmp: Optional[bool] = False,
                 transform: Optional[Callable] = None,
                 overlap: Optional[int] = 0
                 ):
        """
        Args:
            path: The path of the video
            batch_size: len(result) = batch_size
            fps: Extract frames by fps. The parameter 'fps' and 'total' is mutually exclusive
            total: Extract frames by a fix number. The parameter 'fps' and 'total' is mutually exclusive
            tmp_path: Path of temporary file(s).
            transform: A Callable function that apply transformation on each [3, H, W] images.
            overlap: Overlap of two adjacent batches.
        Returns:
            Tuple of (batch, times, indices)
            batch: a list of collected features
            times: the corresponding timestamp of the above features in milliseconds.
            indices: the corresponding indices of the above features. start from zero.
        """
        # sanity check & save properties
        assert type(batch_size) is int and batch_size > 0
        assert type(overlap) is int and 0 <= overlap < batch_size
        self.batch_size = batch_size
        self.transform = transform
        self.overlap = overlap
        self.keep_tmp = keep_tmp
        self.have_generated_tmp_file = False

        if fps is not None and total is not None:
            raise ValueError(f"You can only choose one frame extracting method."
                             f" The parameter 'fps' and 'total' is mutually exclusive")
        elif fps is not None:  # new fps
            self.path = reencode_video_with_diff_fps(path, tmp_path=tmp_path, extraction_fps=fps)
            self.have_generated_tmp_file = True
            for k, v in self._get_video_prop(self.path).items():
                self.__setattr__(k, v)
            # self.fps = fps
            # new_cap = cv2.VideoCapture(self.path)
            # self.num_frames = int(new_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # self.height, self.width = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # new_cap.release()
        elif total is not None:  # fix number of frames
            # ori_cap = cv2.VideoCapture(path)
            # ori_num_frames = ori_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # ori_fps = ori_cap.get(cv2.CAP_PROP_FPS)
            # self.height, self.width = int(ori_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(ori_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # ori_cap.release()
            # self.fps = total * ori_fps / ori_num_frames
            video_prop = self._get_video_prop(path)
            self.height, self.width = video_prop['height'], video_prop['width']
            self.num_frames = total
            self.fps = total * video_prop['fps'] / video_prop['num_frames']
            self.path = reencode_video_with_diff_fps(path, tmp_path=tmp_path, extraction_fps=self.fps)
            self.have_generated_tmp_file = True
        else:  # old fps
            for k, v in self._get_video_prop(path).items():
                self.__setattr__(k, v)
            self.path = path

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.path)
        # sometimes when the target fps is 1 or 2, the first frame of the reencoded video is missing
        # and cap.read returns None but the rest of the frames are ok. timestep is 0.0 for the 2nd frame in
        # this case
        frame_exists, _ = self.cap.read()
        if frame_exists:  # if not missing, go back to the start
            # print("not missing")  # For debug
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print('Detect missing frame')  # For debug
        return self

    def __next__(self) -> Tuple[List[Union[np.ndarray, Tensor]], List[float], List[int]]:
        if not self.cap.isOpened():
            raise StopIteration
        frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx == len(self):
            raise StopIteration

        # Deal with overlap
        if frame_idx - self.overlap >= 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - self.overlap)

        # Normally, this will perform batch_size times.
        # If the read finishes before the batch is filled, a smaller batch will be returned,
        # and the VideoCapture object will be released.
        # If the VideoCapture object is released, will raise StopIteration.
        # If no frame is read, raise StopIteration
        batch, times, indices = [], [], []
        while len(batch) < self.batch_size:
            frame_exists, rgb = self.cap.read()
            if frame_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # start from 0
                # timestamps_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamps_ms = idx / self.fps * 1000
                indices.append(idx)
                times.append(timestamps_ms)
                if self.transform is not None:
                    batch.append(self.transform(rgb))
                else:
                    batch.append(rgb)
            else:
                self.cap.release()
                break
        if len(batch) == 0:
            raise StopIteration

        return batch, times, indices

    def __len__(self):
        return self.num_frames

    def __del__(self):
        # use `hasattr` in case the attribution has not been defined
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'have_generated_tmp_file') and hasattr(self, 'keep_tmp'):
            if self.have_generated_tmp_file and not self.keep_tmp:
                os.remove(self.path)

    @staticmethod
    def _get_video_prop(path):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return dict(fps=fps, num_frames=num_frames, height=height, width=width)
