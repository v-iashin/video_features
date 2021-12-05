import os
# import traceback
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch
from models.i3d.i3d_src.i3d_net import I3D
from models.i3d.transforms.transforms import (Clamp, PermuteAndUnsqueeze,
                                              PILToTensor, ResizeImproved,
                                              ScaleTo1_1, TensorCenterCrop,
                                              ToFloat, ToUInt8)
from models.raft.raft_src.raft import RAFT, InputPadder
from torchvision import transforms
from tqdm import tqdm
from utils.utils import (action_on_extraction, form_list_from_user_input,
                         reencode_video_with_diff_fps,
                         show_predictions_on_dataset)

PWC_MODEL_PATH = './models/pwc/checkpoints/pwc_net_sintel.pt'
RAFT_MODEL_PATH = './models/raft/checkpoints/raft-sintel.pth'
I3D_RGB_PATH = './models/i3d/checkpoints/i3d_rgb.pt'
I3D_FLOW_PATH = './models/i3d/checkpoints/i3d_flow.pt'
PRE_CENTRAL_CROP_MIN_SIDE_SIZE = 256
CENTRAL_CROP_MIN_SIDE_SIZE = 224
DEFAULT_I3D_STEP_SIZE = 64
DEFAULT_I3D_STACK_SIZE = 64
I3D_CLASSES_NUM = 400

class ExtractI3D(torch.nn.Module):

    def __init__(self, args):
        super(ExtractI3D, self).__init__()
        self.feature_type = args.feature_type
        if args.streams is None:
            self.streams = ['rgb', 'flow']
        else:
            self.streams = [args.streams]
        self.path_list = form_list_from_user_input(args)
        self.flow_type = args.flow_type
        self.flow_model_paths = {'pwc': PWC_MODEL_PATH, 'raft': RAFT_MODEL_PATH}
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
            transforms.ToPILImage(),
            ResizeImproved(self.min_side_size),
            PILToTensor(),
            ToFloat(),
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
        flow_xtr_model, models = self.load_model(device)

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

    def extract(self,
                device: torch.device,
                flow_xtr_model: torch.nn.Module,
                models: Dict[str, torch.nn.Module],
                video_path: Union[str, None] = None
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
        def _run_on_a_stack(feats_dict, rgb_stack, models, device, stack_counter, padder=None):
            rgb_stack = torch.cat(rgb_stack).to(device)

            for stream in self.streams:
                with torch.no_grad():
                    # if i3d stream is flow, we first need to calculate optical flow, otherwise, we use rgb
                    # `end_idx-1` and `start_idx+1` because flow is calculated between f and f+1 frames
                    # we also use `end_idx-1` for stream == 'rgb' case: just to make sure the feature length
                    # is same regardless of whether only rgb is used or flow
                    if stream == 'flow':
                        if self.flow_type == 'raft':
                            stream_slice = flow_xtr_model(padder.pad(rgb_stack)[:-1], padder.pad(rgb_stack)[1:])
                        elif self.flow_type == 'pwc':
                            stream_slice = flow_xtr_model(rgb_stack[:-1], rgb_stack[1:])
                        else:
                            raise NotImplementedError
                    elif stream == 'rgb':
                        stream_slice = rgb_stack[:-1]
                    else:
                        raise NotImplementedError
                    # apply transforms depending on the stream (flow or rgb)
                    stream_slice = self.i3d_transforms[stream](stream_slice)
                    # extract features for a stream
                    feats = models[stream](stream_slice, features=True)  # (B, 1024)
                    # add features to the output dict
                    feats_dict[stream].extend(feats.tolist())
                    # show predictions on a daataset
                    if self.show_pred:
                        softmaxes, logits = models[stream](stream_slice, features=False)  # (B, classes=400)
                        print(f'{video_path} @ stack {stack_counter} ({stream} stream)')
                        show_predictions_on_dataset(logits, 'kinetics')

        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # timestamp when the last frame in the stack begins (when the old frame of the last pair ends)
        timestamps_ms = []
        stack = []
        feats_dict = {stream: [] for stream in self.streams}

        # sometimes when the target fps is 1 or 2, the first frame of the reencoded video is missing
        # and cap.read returns None but the rest of the frames are ok. timestep is 0.0 for the 2nd frame in
        # this case
        first_frame = True
        padder = None
        stack_counter = 0
        while cap.isOpened():
            frame_exists, rgb = cap.read()

            if first_frame:
                first_frame = False
                if frame_exists is False:
                    continue

            if frame_exists:
                # preprocess the image
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = self.resize_transforms(rgb)
                rgb = rgb.unsqueeze(0)

                if self.flow_type == 'raft' and padder is None:
                    padder = InputPadder(rgb.shape)

                stack.append(rgb)

                # - 1 is used because we need B+1 frames to calculate B frames
                if len(stack) - 1 == self.stack_size:
                    _run_on_a_stack(feats_dict, stack, models, device, stack_counter, padder)
                    # leaving the elements if step_size < stack_size so they will not be loaded again
                    # if step_size == stack_size one element is left because the flow between the last element
                    # in the prev list and the first element in the current list
                    stack = stack[self.step_size:]
                    stack_counter += 1
                    timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            else:
                # we don't run inference if the stack is not full (applicable for i3d)
                cap.release()
                break

        # removes the video with different fps if it was created to preserve disk space
        if (self.extraction_fps is not None) and (not self.keep_tmp_files):
            os.remove(video_path)

        # transforms list of features into a np array
        feats_dict = {stream: np.array(feats) for stream, feats in feats_dict.items()}
        # also include the timestamps and fps
        feats_dict['fps'] = np.array(fps)
        feats_dict['timestamps_ms'] = np.array(timestamps_ms)

        return feats_dict

    def load_model(self, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, torch.nn.Module]]:
        '''Defines the models, loads checkpoints, sends them to the device.

        Args:
            device (torch.device): The device

        Raises:
            NotImplementedError: if flow type is not implemented.

        Returns:
            Tuple[torch.nn.Module, Dict[str, torch.nn.Module]]: flow extraction module and the model.
        '''
        # Flow extraction module
        if self.flow_type == 'pwc':
            from models.pwc.pwc_src.pwc_net import PWCNet
            flow_xtr_model = PWCNet()
        elif self.flow_type == 'raft':
            flow_xtr_model = RAFT()
            flow_xtr_model = torch.nn.DataParallel(flow_xtr_model, device_ids=[device])
        else:
            raise NotImplementedError

        flow_xtr_model.load_state_dict(torch.load(self.flow_model_paths[self.flow_type], map_location='cpu'))
        flow_xtr_model = flow_xtr_model.to(device)
        flow_xtr_model.eval()

        # Feature extraction models (rgb and flow streams)
        i3d_stream_models = {}
        for stream in self.streams:
            i3d_stream_model = I3D(num_classes=self.i3d_classes_num, modality=stream)
            i3d_stream_model.load_state_dict(torch.load(self.i3d_weights_paths[stream], map_location='cpu'))
            i3d_stream_model = i3d_stream_model.to(device)
            i3d_stream_model.eval()
            i3d_stream_models[stream] = i3d_stream_model

        return flow_xtr_model, i3d_stream_models
