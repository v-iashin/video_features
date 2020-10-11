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
I3D_CLASSES_NUM = 400

class ExtractI3D(torch.nn.Module):

    def __init__(self, args):
        super(ExtractI3D, self).__init__()
        self.feature_type = args.feature_type
        if args.streams is None:
            self.streams = ['rgb', 'flow']
        else:
            self.streams = args.streams
        self.path_list = form_list_from_user_input(args)
        self.pwc_path = PWC_PATH
        self.i3d_weights_paths = {'rgb': I3D_RGB_PATH, 'flow': I3D_FLOW_PATH}
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
        self.i3d_transforms = {
            'rgb': transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ]),
            'flow': transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                Clamp(-20, 20),
                ToUInt8(),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ])
        }
        self.show_pred = args.show_pred
        self.i3d_classes_num = I3D_CLASSES_NUM
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

        flow_xtr_model = PWCNet(self.pwc_path).to(device)
        models = {}
        for stream in self.streams:
            models[stream] = I3D(num_classes=self.i3d_classes_num, modality=stream).to(device).eval()
            models[stream].load_state_dict(torch.load(self.i3d_weights_paths[stream]))

        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                feats_dict = self.extract(device, flow_xtr_model, models, self.path_list[idx])
                action_on_extraction(feats_dict, self.path_list[idx], self.output_path, self.on_extraction)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # traceback.print_exc()  # for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]}. Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, device: torch.device, flow_xtr_model: torch.nn.Module,
                models: Dict[str, torch.nn.Module], video_path: Union[str, None] = None
                ) -> Dict[str, Union[torch.nn.Module, str]]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            flow_xtr_model {torch.nn.Module}
            models {Dict[str, torch.nn.Module]}

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model"-fashion (default: {None})

        Returns:
            Dict[str, Union[torch.nn.Module, str]] -- dict with i3d features and their type
        '''
        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # loading a video rgb: (f, h, w, c)
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

            for stream in self.streams:
                with torch.no_grad():
                    # if i3d stream is flow, we first need to calculate optical flow, otherwise, we use rgb
                    # `end_idx-1` and `start_idx+1` because flow is calculated between f and f+1 frames
                    # we also use `end_idx-1` for stream == 'rgb' case: just to make sure the feature length
                    # is same regardless of whether only rgb is used or flow
                    if stream == 'flow':
                        stream_slice = flow_xtr_model(rgb_slice[:-1], rgb_slice[1:], device)
                    elif stream == 'rgb':
                        stream_slice = rgb_slice[:-1]
                    else:
                        raise NotImplementedError
                    stream_slice = self.i3d_transforms[stream](stream_slice)
                    feats = models[stream](stream_slice, features=True)  # (B, 1024)
                    feats_dict[stream].extend(feats.tolist())
                    if self.show_pred:
                        softmaxes, logits = models[stream](stream_slice, features=False)  # (B, classes=400)
                        print(f'{video_path} @ stack {stack_idx} ({stream} stream)')
                        show_predictions_on_dataset(logits, 'kinetics')

        # removes the video with different fps if it was created to preserve disk space
        if (self.extraction_fps is not None) and (not self.keep_tmp_files):
            os.remove(video_path)

        # transforms list of features into a np array
        feats_dict = {stream: np.array(feats) for stream, feats in feats_dict.items()}

        return feats_dict
