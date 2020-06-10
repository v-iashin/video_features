from typing import Dict, List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from models.i3d.utils.utils import center_crop


def i3d_features(frame_paths: List[str], stack_size: int, device: torch.device,
                 pwc_net: torch.nn.Module, i3d_net: torch.nn.Module, show_i3d_preds: bool = None,
                 classes_path: bool = None, features: Union[None, str] = 'separately_rgb_flow'
                 ) -> Dict[str, Union[str, torch.FloatTensor]]:
    '''Extracts i3d features using different options. Also inputs models, so they are not made here
    every time.

    Arguments:
        frame_paths {List[str]} -- list of paths
        stack_size {int} -- feature time span (how many frames i3d will process to form one feature)
        device {torch.device}
        pwc_net {torch.nn.Module} -- pwc model
        i3d_net {torch.nn.Module} -- i3d model

    Keyword Arguments:
        show_i3d_preds {bool} -- to show the predictions of i3d on Kinetics (default: {None})
        classes_path {bool} -- Labels. If show_i3d_preds, this should be specified (default: {None})
        features {Union[None, str]} -- tells i3d net which features to output
                                       (default: {'separately_rgb_flow'})

    Returns:
        Dict[str, Union[str, torch.FloatTensor]] -- dict features and the type (specified in 
                                                    'features')
    '''
    # before we start to extract features, we save the resolution of the video
    W, H = Image.open(frame_paths[0][0]).size

    i3d_feats = {}
    i3d_feats['type'] = str(features)

    if features == 'separately_rgb_flow':
        i3d_feats['rgb'] = torch.zeros(len(frame_paths), 1024, device=device)
        i3d_feats['flow'] = torch.zeros(len(frame_paths), 1024, device=device)
    else:
        i3d_feats[str(features)] = torch.zeros(len(frame_paths), 1024, device=device)

    for stack_idx, frame_path_stack in enumerate(frame_paths):
        # T+1, since T flow frames require T+1 rgb frames
        rgb_stack = torch.zeros(stack_size+1, 3, H, W, device=device)

        # load the rgb stack
        for frame_idx, frame_path in enumerate(frame_path_stack):
            rgb = cv2.imread(frame_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.array(rgb).transpose(2, 0, 1)
            rgb = torch.FloatTensor(rgb).unsqueeze(0)
            rgb_stack[frame_idx] = rgb

        # calculate the optical flow
        with torch.no_grad():
            flow_stack = pwc_net(rgb_stack[:-1], rgb_stack[1:], device)

        # crop
        rgb_stack = center_crop(rgb_stack[:-1], crop_size=224)
        flow_stack = center_crop(flow_stack, crop_size=224)
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

        # extract i3d features (show predictions or output features)
        if show_i3d_preds:
            with torch.no_grad():
                softmaxes, logits = i3d_net(rgb_stack, flow_stack, features=None)

            top_val, top_idx = torch.sort(softmaxes, 1, descending=True)
            print('===== Final predictions =====')
            kinetics_classes = [x.strip() for x in open(classes_path)]
            top_k = 5
            print('logits proba class ')
            for i in range(top_k):
                logits_score = logits[0, top_idx[0, i]].data.item()
                print(f'{logits_score:.3f} {top_val[0, i]:.3f} {kinetics_classes[top_idx[0, i]]}')
        else:
            with torch.no_grad():
                if features == 'separately_rgb_flow':
                    i3d_feats['rgb'][stack_idx], i3d_feats['flow'][stack_idx] = i3d_net(
                        rgb_stack, flow_stack, features
                    )
                else:
                    i3d_feats[str(features)][stack_idx] = i3d_net(rgb_stack, flow_stack, features)

    return i3d_feats
