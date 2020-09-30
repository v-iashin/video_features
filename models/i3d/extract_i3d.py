import os
import pathlib
import shutil
import subprocess
# import traceback
from typing import Dict, Union

import cv2
import numpy as np
import torch
from models.i3d.flow_src.pwc_net import PWCNet
from models.i3d.i3d_src.i3d_net import I3D_RGB_FLOW
from PIL import Image
from tqdm import tqdm

from utils.utils import (action_on_extraction, form_iter_list,
                         form_list_from_user_input,
                         show_predictions_on_dataset, which_ffmpeg)

PWC_PATH = './models/i3d/checkpoints/pwc_net_sintel.pt'
I3D_RGB_PATH = './models/i3d/checkpoints/i3d_rgb.pt'
I3D_FLOW_PATH = './models/i3d/checkpoints/i3d_flow.pt'
PRE_CENTRAL_CROP_MIN_SIDE_SIZE = 256
CENTRAL_CROP_MIN_SIDE_SIZE = 224
DEFAULT_I3D_STEP_SIZE = 64
DEFAULT_I3D_STACK_SIZE = 64
I3D_FEATURE_TYPE = 'separately_rgb_flow'


class ExtractI3D(torch.nn.Module):

    def __init__(self, args):
        super(ExtractI3D, self).__init__()
        self.feature_type = args.feature_type
        self.path_list = form_list_from_user_input(args)
        self.pwc_path = PWC_PATH
        self.i3d_rgb_path = I3D_RGB_PATH
        self.i3d_flow_path = I3D_FLOW_PATH
        self.min_side_size = PRE_CENTRAL_CROP_MIN_SIDE_SIZE
        self.central_crop_size = CENTRAL_CROP_MIN_SIDE_SIZE
        self.extraction_fps = args.extraction_fps
        self.step_size = args.step_size
        self.stack_size = args.stack_size
        if self.step_size is None:
            self.step_size = DEFAULT_I3D_STEP_SIZE
        if self.stack_size is None:
            self.stack_size = DEFAULT_I3D_STACK_SIZE
        self.show_class_pred = args.show_class_pred
        self.keep_tmp_files = args.keep_tmp_files
        self.on_extraction = args.on_extraction
        self.tmp_path = os.path.join(args.tmp_path, self.feature_type)
        self.output_path = os.path.join(args.output_path, self.feature_type)
        self.progress = tqdm(total=len(self.path_list))

    def forward(self, indices: torch.LongTensor):
        '''
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        '''
        device = indices.device

        pwc_model = PWCNet(self.pwc_path).to(device)
        i3d_model = I3D_RGB_FLOW(self.i3d_rgb_path, self.i3d_flow_path).to(device)

        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                feats_dict = self.extract(device, pwc_model, i3d_model, self.path_list[idx])
                action_on_extraction(feats_dict, self.path_list[idx], self.output_path, self.on_extraction)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # traceback.print_exc()  # for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]}. Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, device: torch.device, pwc_model: torch.nn.Module, i3d_model: torch.nn.Module,
                video_path: Union[str, None] = None
                ) -> Dict[str, Union[torch.nn.Module, str]]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            pwc_model {torch.nn.Module}
            i3d_model {torch.nn.Module}

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model"-fashion (default: {None})

        Returns:
            Dict[str, Union[torch.nn.Module, str]] -- dict with i3d features and their type
        '''
        frames_dir = self.extract_frames_from_video(video_path)
        # sorted list of frame paths
        frame_paths = [os.path.join(frames_dir, fname) for fname in sorted(os.listdir(frames_dir))]
        # T+1, since T flow frames require T+1 rgb frames
        frame_paths = form_iter_list(frame_paths, self.step_size, self.stack_size+1)

        # before we start to extract features, we save the resolution of the video
        W, H = Image.open(frame_paths[0][0]).size
        video_path = pathlib.Path(frame_paths[0][0]).parent

        i3d_rgb = []
        i3d_flow = []

        for stack_idx, frame_path_stack in enumerate(frame_paths):
            # T+1, since T flow frames require T+1 rgb frames
            rgb_stack = torch.zeros(self.stack_size+1, 3, H, W, device=device)

            # load the rgb stack
            for frame_idx, frame_path in enumerate(frame_path_stack):
                rgb = cv2.imread(frame_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = np.array(rgb).transpose(2, 0, 1)
                rgb = torch.FloatTensor(rgb).unsqueeze(0)
                rgb_stack[frame_idx] = rgb

            # calculate the optical flow
            with torch.no_grad():
                flow_stack = pwc_model(rgb_stack[:-1], rgb_stack[1:], device)

            # crop
            rgb_stack = self.center_crop(rgb_stack[:-1], crop_size=self.central_crop_size)
            flow_stack = self.center_crop(flow_stack, crop_size=self.central_crop_size)
            # scaling values to be between [-1, 1]
            rgb_stack = (2 * rgb_stack / 255) - 1
            # clamping
            flow_stack = torch.clamp(flow_stack, min=-20, max=20)
            # preprocessing as in
            # https://github.com/deepmind/kinetics-i3d/issues/61#issuecomment-506727158
            # but for pytorch
            # [-20, 20] -> [0, 255]
            flow_stack = 128 + 255 / 40 * flow_stack
            # make it an integer
            flow_stack = flow_stack.round()
            # [0, 255] -> [-1, 1]
            flow_stack = (2 * flow_stack / 255) - 1

            # form inputs to I3D (RGB + FLOW)
            rgb_stack.unsqueeze_(0), flow_stack.unsqueeze_(0)
            rgb_stack = rgb_stack.permute(0, 2, 1, 3, 4)
            flow_stack = flow_stack.permute(0, 2, 1, 3, 4)

            # extract i3d features
            with torch.no_grad():
                feat_rgb, feat_flow = i3d_model(rgb_stack, flow_stack, features='separately_rgb_flow')
                i3d_rgb.extend(feat_rgb.tolist())
                i3d_flow.extend(feat_flow.tolist())

                if self.show_class_pred:
                    softmaxes, logits = i3d_model(rgb_stack, flow_stack, features=None)
                    print(f'{video_path} @ stack {stack_idx}')
                    show_predictions_on_dataset(logits, 'kinetics')

        # removes the folder with extracted frames to preserve disk space
        if not self.keep_tmp_files:
            shutil.rmtree(frames_dir)

        feats_dict = {
            'rgb': np.array(i3d_rgb),
            'flow': np.array(i3d_flow)
        }

        return feats_dict

    def center_crop(self, tensor: torch.FloatTensor, crop_size: int = 224) -> torch.FloatTensor:
        '''
            Takes a tensor of input (..., H, W) and performs a
            center crop of provided size.
        '''
        H, W = tensor.size(-2), tensor.size(-1)
        from_H = ((H - crop_size) // 2)
        from_W = ((W - crop_size) // 2)
        to_H = from_H + crop_size
        to_W = from_W + crop_size
        return tensor[..., from_H:to_H, from_W:to_W]

    def extract_frames_from_video(self, video_path: str) -> str:
        '''Extracts frames from a video using the specified path.

        Arguments:
            video_path {str} -- path to a video

        Returns:
            [str] -- path to the folder with extracted frames
        '''
        assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
        assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'

        video = cv2.VideoCapture(video_path)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)

        # if extraction fps is not specified, use the original fps; if specified, use it
        if self.extraction_fps is None:
            extraction_fps = video.get(cv2.CAP_PROP_FPS)
        else:
            extraction_fps = self.extraction_fps

        # Make dir for frames
        folder_name = f'{pathlib.Path(video_path).stem}_frames'
        frames_path = os.path.join(self.tmp_path, folder_name)

        if os.path.exists(frames_path):
            print(f'Warning: folder {frames_path} already exists. Possibly these frames are going to be bad')
        else:
            os.makedirs(frames_path, exist_ok=True)

        # vertical/horizontal video handling
        if width > height:
            size = f'-1:{self.min_side_size}'
        else:
            size = f'{self.min_side_size}:-1'

        # Extract frames: call ffmpeg
        extract_cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic -y -i {video_path}'
        extract_cmd += f' -vf scale={size} -r {extraction_fps} -q:v 2 {frames_path}/%6d.jpg'

        subprocess.call(extract_cmd.split())

        return frames_path
