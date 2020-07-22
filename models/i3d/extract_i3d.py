# import argparse
import os
import shutil
# import sys
from typing import Dict, Union

import numpy as np
import torch
from tqdm import tqdm
# import traceback

# sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models.i3d.flow_src.pwc_net import PWCNet
from models.i3d.i3d_src.i3d_feats import i3d_features
from models.i3d.i3d_src.i3d_net import I3D_RGB_FLOW
from models.i3d.utils.utils import extract_frames_from_video, form_iter_list
from utils.utils import form_list_from_user_input

KINETICS_CLASSES_PATH = './models/i3d/checkpoints/label_map.txt'
PWC_PATH = './models/i3d/checkpoints/pwc_net.pt'
I3D_RGB_PATH = './models/i3d/checkpoints/i3d_rgb.pt'
I3D_FLOW_PATH = './models/i3d/checkpoints/i3d_flow.pt'
PRE_CENTRAL_CROP_MIN_SIDE_SIZE = 256
CENTRAL_CROP_MIN_SIDE_SIZE = 224
DEFAULT_I3D_STEP_SIZE = 64
DEFAULT_I3D_STACK_SIZE = 64


class ExtractI3D(torch.nn.Module):

    def __init__(self, args):
        super(ExtractI3D, self).__init__()
        self.path_list = form_list_from_user_input(args)
        self.pwc_path = PWC_PATH
        self.i3d_rgb_path = I3D_RGB_PATH
        self.i3d_flow_path = I3D_FLOW_PATH
        self.min_side_size = PRE_CENTRAL_CROP_MIN_SIDE_SIZE
        self.extraction_fps = args.extraction_fps
        self.step_size = args.step_size
        self.stack_size = args.stack_size
        if self.step_size is None:
            self.step_size = DEFAULT_I3D_STEP_SIZE
        if self.stack_size is None:
            self.stack_size = DEFAULT_I3D_STACK_SIZE
        self.show_kinetics_pred = args.show_kinetics_pred
        self.kinetics_class_path = KINETICS_CLASSES_PATH
        self.keep_frames = args.keep_frames
        self.on_extraction = args.on_extraction
        self.tmp_path = args.tmp_path
        self.output_path = args.output_path
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
                self.extract(device, pwc_model, i3d_model, idx)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # traceback.print_exc()  # for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]}. Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, device: torch.device, pwc_model: torch.nn.Module, i3d_model: torch.nn.Module, 
                idx: int, video_path: Union[str, None] = None
                ) -> Dict[str, Union[torch.nn.Module, str]]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            pwc_model {torch.nn.Module}
            i3d_model {torch.nn.Module}
            idx {int} -- index to self.video_paths

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as 
                                             "path -> i3d features"-fashion (default: {None})

        Returns:
            Dict[str, Union[torch.nn.Module, str]] -- dict with i3d features and their type
        '''
        if video_path is None:
            video_path = self.path_list[idx]

        frames_dir = extract_frames_from_video(video_path, self.extraction_fps, 
                                               self.min_side_size, self.tmp_path)
        # sorted list of frame paths
        frame_paths = [os.path.join(frames_dir, fname) for fname in sorted(os.listdir(frames_dir))]
        # T+1, since T flow frames require T+1 rgb frames
        frame_paths = form_iter_list(frame_paths, self.step_size, self.stack_size+1)
        # extract features
        i3d_feats = i3d_features(frame_paths, self.stack_size, device, pwc_model, i3d_model, 
                                 self.show_kinetics_pred, self.kinetics_class_path)

        # removes the folder with extracted frames to preserve disk space
        if not self.keep_frames:
            shutil.rmtree(frames_dir)

        # What to do once features are extracted.
        if self.on_extraction == 'print':
            print(i3d_feats)
        elif self.on_extraction == 'save_numpy':
            # make dir if doesn't exist
            os.makedirs(self.output_path, exist_ok=True)
            # extract file name and change the extention
            filename_rgb = os.path.split(video_path)[-1].replace('.mp4', '_rgb.npy')
            filename_flow = os.path.split(video_path)[-1].replace('.mp4', '_flow.npy')
            # construct the paths to save the features
            feature_rgb_path = os.path.join(self.output_path, filename_rgb)
            feature_flow_path = os.path.join(self.output_path, filename_flow)
            # save features
            np.save(feature_rgb_path, i3d_feats['rgb'].cpu())
            np.save(feature_flow_path, i3d_feats['flow'].cpu())
        else:
            raise NotImplementedError

        return i3d_feats
