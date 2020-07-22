import os
import numpy as np
import torch
from tqdm import tqdm
# import traceback
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import Compose
import models.r21d.transforms.rgb_transforms as T
from models.r21d.r21d_src.r21d_feats import r21d_features
from utils.utils import form_list_from_user_input
from typing import Dict, Union

PRE_CENTRAL_CROP_SIZE = (128, 171)
CENTRAL_CROP_MIN_SIDE_SIZE = 112
DEFAULT_R21D_STEP_SIZE = 16
DEFAULT_R21D_STACK_SIZE = 16
KINETICS_CLASS_LABELS = './models/r21d/checkpoints/label_map.txt'
R21D_PATH = './models/r21d/checkpoints/r21d_rgb.pt'
R21D_FEATURE_SIZE = 512

class ExtractR21D(torch.nn.Module):

    def __init__(self, args):
        super(ExtractR21D, self).__init__()
        self.path_list = form_list_from_user_input(args)
        self.r21d_path = R21D_PATH
        self.central_crop_min_side_size = CENTRAL_CROP_MIN_SIDE_SIZE
        self.extraction_fps = args.extraction_fps
        self.r21d_feature_size = R21D_FEATURE_SIZE
        self.step_size = args.step_size
        self.stack_size = args.stack_size
        if self.step_size is None:
            self.step_size = DEFAULT_R21D_STEP_SIZE
        if self.stack_size is None:
            self.stack_size = DEFAULT_R21D_STACK_SIZE
        self.transforms = Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize(PRE_CENTRAL_CROP_SIZE),
            T.Normalize(mean=T.KINETICS_MEAN, std=T.KINETICS_STD),
            T.CenterCrop((CENTRAL_CROP_MIN_SIDE_SIZE, CENTRAL_CROP_MIN_SIDE_SIZE))
        ])
        self.show_kinetics_pred = args.show_kinetics_pred
        self.kinetics_class_labels = KINETICS_CLASS_LABELS
        self.keep_frames = args.keep_frames  # not used, create an issue if you would like to save the frames
        self.on_extraction = args.on_extraction
        self.tmp_path = args.tmp_path  # not used, create an issue if you would like to save the frames
        self.output_path = args.output_path
        self.progress = tqdm(total=len(self.path_list))

    def forward(self, indices: torch.LongTensor):
        '''
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        '''
        device = indices.device
        
        r21d = r2plus1d_18(pretrained=True).to(device)
        r21d.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        r21d_class = r21d.fc
        r21d.fc = torch.nn.Identity()

        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                self.extract(device, r21d, r21d_class, idx)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # prints only the last line of an error. Use `traceback.print_exc()` for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]} with error (↑). Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, device: torch.device, model: torch.nn.Module, classifier: torch.nn.Module,
                idx: int, video_path: Union[str, None] = None
                ) -> Dict[str, Union[torch.nn.Module, str]]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            model {torch.nn.Module}
            classifier {torch.nn.Module} -- pre-trained classification layer, will be used if 
                                            show_kinetics_pred is True
            idx {int} -- index to self.video_paths

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as 
                                             "path -> i3d features"-fashion (default: {None})

        Returns:
            Dict[str, Union[torch.nn.Module, str]] -- dict with i3d features and their type
        '''
        if video_path is None:
            video_path = self.path_list[idx]

        # extract features
        r21d_feats = r21d_features(
            model, video_path, self.r21d_feature_size, self.stack_size, self.step_size, device,
            self.transforms, self.show_kinetics_pred, self.kinetics_class_labels, classifier
        )

        # What to do once features are extracted.
        if self.on_extraction == 'print':
            print(r21d_feats)
        elif self.on_extraction == 'save_numpy':
            # make dir if doesn't exist
            os.makedirs(self.output_path, exist_ok=True)
            # extract file name and change the extention
            filename_rgb = os.path.split(video_path)[-1].replace('.mp4', '_rgb.npy')
            # construct the paths to save the features
            feature_rgb_path = os.path.join(self.output_path, filename_rgb)
            # save features
            np.save(feature_rgb_path, r21d_feats.cpu())
        else:
            raise NotImplementedError

        return r21d_feats

