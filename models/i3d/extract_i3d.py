import os
# import traceback
from typing import Dict, Union

import numpy as np
import torch
from models.i3d.pwc_src.pwc_net import PWCNet
from models.i3d.i3d_src.i3d_net import I3D
from models.i3d.transforms.transforms import (Clamp, PermuteAndUnsqueeze,
                                              Resize, ScaleTo1_1,
                                              TensorCenterCrop, ToCFHW_ToFloat,
                                              ToFCHW, ToUInt8)
from torchvision import transforms
from torchvision.io import read_video
from tqdm import tqdm
from utils.utils import (action_on_extraction, form_list_from_user_input,
                         form_slices, reencode_video_with_diff_fps,
                         show_predictions_on_dataset)

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
        self.streams = args.streams
        self.path_list = form_list_from_user_input(args)
        self.pwc_path = PWC_PATH
        self.rgb_weights_path = I3D_RGB_PATH
        self.flow_weights_path = I3D_FLOW_PATH
        self.min_side_size = PRE_CENTRAL_CROP_MIN_SIDE_SIZE
        self.central_crop_size = CENTRAL_CROP_MIN_SIDE_SIZE
        self.extraction_fps = args.extraction_fps
        self.step_size = args.step_size
        self.stack_size = args.stack_size
        if self.step_size is None:
            self.step_size = DEFAULT_I3D_STEP_SIZE
        if self.stack_size is None:
            self.stack_size = DEFAULT_I3D_STACK_SIZE
        self.resize_transforms = transforms.Compose([
            ToCFHW_ToFloat(),
            Resize(self.min_side_size),
            ToFCHW()
        ])
        self.i3d_rgb_transforms = transforms.Compose([
            TensorCenterCrop(self.central_crop_size),
            ScaleTo1_1(),
            PermuteAndUnsqueeze()
        ])
        self.i3d_flow_transforms = transforms.Compose([
            TensorCenterCrop(self.central_crop_size),
            Clamp(-20, 20),
            ToUInt8(),
            ScaleTo1_1(),
            PermuteAndUnsqueeze()
        ])
        self.show_pred = args.show_pred
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
        # i3d_model = I3D_RGB_FLOW(self.i3d_rgb_path, self.i3d_flow_path).to(device)
        i3d_rgb_model = None
        i3d_flow_model = None
        if 'rgb' in self.streams:
            i3d_rgb_model = I3D(num_classes=400, modality='rgb').to(device)
            i3d_rgb_model.load_state_dict(torch.load(self.rgb_weights_path))
            i3d_rgb_model.eval()
        if 'flow' in self.streams:
            i3d_flow_model = I3D(num_classes=400, modality='flow').to(device)
            i3d_flow_model.load_state_dict(torch.load(self.flow_weights_path))
            i3d_flow_model.eval()

        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                feats_dict = self.extract(device, pwc_model, i3d_rgb_model, i3d_flow_model, self.path_list[idx])
                action_on_extraction(feats_dict, self.path_list[idx], self.output_path, self.on_extraction)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # traceback.print_exc()  # for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]}. Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, device: torch.device, pwc_model: torch.nn.Module,
                i3d_rgb_model: torch.nn.Module, i3d_flow_model: torch.nn.Module,
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
        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # rgb: (f, h, w, c)
        video_rgb, _audio, _info = read_video(video_path)

        # sanity check
        if self.extraction_fps is not None and _info['video_fps'] != self.extraction_fps:
            print(f'self.extraction_fps {self.extraction_fps} != file`s fps {_info["video_fps"]}')

        # (f, c, h`, w`)
        video_rgb = self.resize_transforms(video_rgb)
        video_rgb = video_rgb.to(device)

        # form slices which will be used to slice video_rgb, takes into accound the step and stack sizes
        slices = form_slices(video_rgb.size(0), self.stack_size+1, self.step_size)

        # init a dict: {'rgb': [], 'flow': []} or any subset of these
        feats_dict = {stream: [] for stream in self.streams}

        for stack_idx, (start_idx, end_idx) in enumerate(slices):
            # since we apply some transforms on rgb later, we allocate the slice into a dedicated variable
            rgb_slice = video_rgb[start_idx:end_idx]

            if 'flow' in self.streams:
                with torch.no_grad():
                    # `end_idx-1` and `start_idx+1` because flow is calculated between f and f+1 frames
                    flow_slice = pwc_model(rgb_slice[:-1], rgb_slice[1:], device)
                    flow_slice = self.i3d_flow_transforms(flow_slice)
                    # shouldn't reassign variables as flow_slice is also used in show_preds
                    flow_feats = i3d_flow_model(flow_slice, features=True)  # (B, 1024)
                    feats_dict['flow'].extend(flow_feats.tolist())
                    if self.show_pred:
                        softmaxes, logits = i3d_flow_model(flow_slice, features=False)
                        print(f'{video_path} @ stack {stack_idx} (flow stream)')
                        show_predictions_on_dataset(logits, 'kinetics')

            if 'rgb' in self.streams:
                with torch.no_grad():
                    rgb_slice = self.i3d_rgb_transforms(rgb_slice[:-1])
                    # shouldn't reassign variables as rgb_slice is also used in show_preds
                    rgb_feats = i3d_rgb_model(rgb_slice, features=True)
                    feats_dict['rgb'].extend(rgb_feats.tolist())
                    if self.show_pred:
                        softmaxes, logits = i3d_rgb_model(rgb_slice, features=False)
                        print(f'{video_path} @ stack {stack_idx} (rgb stream)')
                        show_predictions_on_dataset(logits, 'kinetics')

        # removes the video with different fps if it was created to preserve disk space
        if (self.extraction_fps is not None) and (not self.keep_tmp_files):
            os.remove(video_path)

        # transforms list of features into a np array
        feats_dict = {stream: np.array(feats) for stream, feats in feats_dict.items()}

        return feats_dict
