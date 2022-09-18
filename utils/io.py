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
        '''
        Args:
            path: The path of the video
            batch_size: len(result) = batch_size
            fps: Extract frames by fps. The parameter 'fps' and 'total' is mutually exclusive
            total: Extract frames by a fix number. The parameter 'fps' and 'total' is mutually exclusive
            tmp_path: Path of temporary file(s).
            keep_tmp: whether keep the temporary file.
            transform: A Callable object that applies transformation on each [3, H, W] images.
            overlap: Overlap of two adjacent batches.
        Returns:
            Tuple of (batch, times, indices)
            batch: a list of collected features
            times: the corresponding timestamp of the above features in milliseconds.
            indices: the corresponding indices of the above features. start from zero.
        '''
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
        elif total is not None:  # fix number of frames
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
        self.current_idx = 0  # maintain the index of current frame instead of getting property in CV2 to avoid bugs
        self._last_batch, self._last_times, self._last_indices = [], [], []  # cache the overlap
        # BUG of cv2:
        # Sometimes frame#0 is missing, which needs to skip.
        frame_exists, _ = self.cap.read()
        if frame_exists:  # if not missing, go back to the start
            self.cap.release()
            self.cap = cv2.VideoCapture(self.path)
        else:
            print('Detect missing frame')  # For debug
        return self

    def __next__(self) -> Tuple[List[Union[np.ndarray, Tensor]], List[float], List[int]]:
        """
        Normally, a call will read `batch_size-overlap` frames from the video and `overlap` frames from the cache.
        As exceptions, the first batch reads `batch_size` frames and the last batch may contain fewer frames.
        """
        if not self.cap.isOpened():
            raise StopIteration
        # If all frames have been read at the beginning, raise StopIteration
        if self.current_idx == len(self):
            raise StopIteration

        # load overlap
        batch, times, indices = [], [], []
        if self.overlap != 0 and self.current_idx != 0:
            batch += self._last_batch
            times += self._last_times
            indices += self._last_indices

        while len(batch) < self.batch_size:
            frame_exists, rgb = self.cap.read()
            self.current_idx += 1
            if frame_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                timestamps_ms = (self.current_idx - 1) / self.fps * 1000
                indices.append(self.current_idx - 1)
                times.append(timestamps_ms)
                if self.transform is not None:
                    batch.append(self.transform(rgb))
                else:
                    batch.append(rgb)
            else:
                # If read a non-exist frame, which indicates all frames of the video have been read,
                # release the VideoCapture and return the smaller batch. The StopIteration will be
                # raised in the next start.
                self.cap.release()
                break
        if len(batch) == 0:
            raise StopIteration

        # save overlap
        if self.overlap != 0:
            self._last_batch = batch[-self.overlap:]
            self._last_times = times[-self.overlap:]
            self._last_indices = indices[-self.overlap:]

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
        """Get properties of a video"""
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return dict(fps=fps, num_frames=num_frames, height=height, width=width)
