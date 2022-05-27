import os
from typing import Dict, Tuple, Union

import models.r21d.transforms.rgb_transforms as T
import numpy as np
import torch
from torchvision.io.video import read_video
# import traceback
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import Compose
from tqdm import tqdm
from utils.utils import (action_on_extraction, form_list_from_user_input,
                         form_slices, reencode_video_with_diff_fps, show_predictions_on_dataset)

PRE_CENTRAL_CROP_SIZE = (128, 171)
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]
CENTRAL_CROP_MIN_SIDE_SIZE = 112
R21D_MODEL_CFG = {
    'r2plus1d_18_16_kinetics': {
        'repo': None,
        'stack_size': 16, 'step_size': 16, 'num_classes': 400, 'dataset': 'kinetics'
    },
    'r2plus1d_34_32_ig65m_ft_kinetics': {
        'repo': 'moabitcoin/ig65m-pytorch', 'model_name_in_repo': 'r2plus1d_34_32_kinetics',
        'stack_size': 32, 'step_size': 32, 'num_classes': 400, 'dataset': 'kinetics'
    },
    'r2plus1d_34_8_ig65m_ft_kinetics': {
        'repo': 'moabitcoin/ig65m-pytorch', 'model_name_in_repo': 'r2plus1d_34_8_kinetics',
        'stack_size': 8, 'step_size': 8, 'num_classes': 400, 'dataset': 'kinetics'
    },
}

class ExtractR21D(torch.nn.Module):

    def __init__(self, args):
        super(ExtractR21D, self).__init__()
        self.feature_type = args.feature_type
        self.model_name = args.model_name
        self.model_def = R21D_MODEL_CFG[self.model_name]
        self.path_list = form_list_from_user_input(args)
        self.central_crop_min_side_size = CENTRAL_CROP_MIN_SIDE_SIZE
        self.extraction_fps = args.extraction_fps
        self.step_size = args.step_size
        self.stack_size = args.stack_size
        if self.step_size is None:
            self.step_size = self.model_def['step_size']
        if self.stack_size is None:
            self.stack_size = self.model_def['stack_size']
        self.transforms = Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize(PRE_CENTRAL_CROP_SIZE),
            T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
            T.CenterCrop((CENTRAL_CROP_MIN_SIDE_SIZE, CENTRAL_CROP_MIN_SIDE_SIZE))
        ])
        self.show_pred = args.show_pred
        self.keep_tmp_files = args.keep_tmp_files
        self.on_extraction = args.on_extraction
        # not used, create an issue if you would like to save the frames
        self.tmp_path = os.path.join(args.tmp_path, self.feature_type)
        self.output_path = os.path.join(args.output_path, self.feature_type)
        self.progress = tqdm(total=len(self.path_list))

    def forward(self, indices: torch.LongTensor):
        '''
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        '''
        device = indices.device
        model, class_head = self.load_model(device)

        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                feats_dict = self.extract(device, model, class_head, self.path_list[idx])
                action_on_extraction(feats_dict, self.path_list[idx], self.output_path, self.on_extraction)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # prints only the last line of an error. Use `traceback.print_exc()` for the whole traceback
                # traceback.print_exc()
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]} with error (↑). Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, device: torch.device, model: torch.nn.Module, classifier: torch.nn.Module,
                video_path: Union[str, None] = None
                ) -> Dict[str, np.ndarray]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            model {torch.nn.Module}
            classifier {torch.nn.Module} -- pre-trained classification layer, will be used if
                                            show_pred is True

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model features"-fashion (default: {None})

        Returns:
            Dict[str, np.ndarray] -- the dict with numpy feature
        '''
        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # read a video
        rgb, audio, info = read_video(video_path, pts_unit='sec')
        # prepare data (first -- transform, then -- unsqueeze)
        rgb = self.transforms(rgb) # could run out of memory here
        rgb = rgb.unsqueeze(0)
        # slice the stack of frames
        slices = form_slices(rgb.size(2), self.stack_size, self.step_size)

        vid_feats = []

        for stack_idx, (start_idx, end_idx) in enumerate(slices):
            # inference
            with torch.no_grad():
                output = model(rgb[:, :, start_idx:end_idx, :, :].to(device))
                vid_feats.extend(output.tolist())

                # show predicitons on kinetics dataset (might be useful for debugging)
                if self.show_pred:
                    logits = classifier(output)
                    dataset_name = self.model_def['dataset']
                    print(f'{video_path} @ frames ({start_idx}, {end_idx})')
                    show_predictions_on_dataset(logits, dataset_name)

        feats_dict = {
            self.feature_type: np.array(vid_feats),
        }

        return feats_dict

    def load_model(self, device: torch.device) -> Tuple[torch.nn.Module]:
        '''Defines the models, loads checkpoints, sends them to the device.

        Args:
            device (torch.device): The device

        Raises:
            NotImplementedError: if flow type is not implemented.

        Returns:
            Tuple[torch.nn.Module]: the model with identity head, the original classifier
        '''
        if self.model_name == 'r2plus1d_18_16_kinetics':
            model = r2plus1d_18(pretrained=True)
        else:
            model = torch.hub.load(
                self.model_def['repo'],
                model=self.model_def['model_name_in_repo'],
                num_classes=self.model_def['num_classes'],
                pretrained=True,
            )
        model = model.to(device)
        model.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        class_head = model.fc
        model.fc = torch.nn.Identity()
        return model, class_head
