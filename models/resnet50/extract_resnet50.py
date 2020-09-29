import os
import pathlib
from models.resnet50.resnet50_src.resnet50_feats import resnet50_features
import numpy as np
import torch
from tqdm import tqdm
# import traceback
import torchvision.models as models
import torchvision.transforms as transforms
from utils.utils import form_list_from_user_input
from typing import Dict, Union

RESIZE_SIZE = 256
CENTER_CROP_SIZE = 224
TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD = [0.229, 0.224, 0.225]
IMAGENET_CLASS_PATH = './models/resnet50/checkpoints/IN_label_map.txt'

class ExtractResNet50(torch.nn.Module):

    def __init__(self, args):
        super(ExtractResNet50, self).__init__()
        self.feature_type = args.feature_type
        self.path_list = form_list_from_user_input(args)
        self.batch_size = args.batch_size
        self.central_crop_size = CENTER_CROP_SIZE
        # todo fix the extraction such that is will resample features according to this value
        self.extraction_fps = args.extraction_fps
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CENTER_CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD)
        ])
        self.show_imagenet_pred = args.show_imagenet_pred
        self.imagenet_class_path = IMAGENET_CLASS_PATH
        self.keep_frames = args.keep_frames
        self.on_extraction = args.on_extraction
        self.tmp_path = args.tmp_path
        self.output_path = os.path.join(args.output_path, self.feature_type)
        self.progress = tqdm(total=len(self.path_list))

    def forward(self, indices: torch.LongTensor):
        '''
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        '''
        device = indices.device

        model = models.resnet50(pretrained=True).to(device)
        model.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        model_class = model.fc
        model.fc = torch.nn.Identity()

        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                self.extract(device, model, model_class, idx)
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
                ) -> Dict[str, np.ndarray]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            model {torch.nn.Module}
            classifier {torch.nn.Module} -- pre-trained classification layer, will be used if
                                            show_imagenet_pred is True
            idx {int} -- index to self.video_paths

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model"-fashion (default: {None})

        Returns:
            Dict[str, np.ndarray]: 'features_nme', 'fps', 'timestamps_ms'
        '''
        if video_path is None:
            video_path = self.path_list[idx]

        # extract features
        feats_dict = resnet50_features(
            model, video_path, self.batch_size, device, self.transforms, self.show_imagenet_pred,
            self.imagenet_class_path, classifier, self.feature_type
        )

        # What to do once features are extracted.
        if self.on_extraction == 'print':
            print(feats_dict)
        elif self.on_extraction == 'save_numpy':
            # make dir if doesn't exist
            os.makedirs(self.output_path, exist_ok=True)
            # since the features are enclosed in a dict with another meta information we will iterate on kv
            for key, value in feats_dict.items():
                # extract file name and change the extention
                fname = f'{pathlib.Path(video_path).stem}_{key}.npy'
                # construct the paths to save the features
                fpath = os.path.join(self.output_path, fname)
                # save the info behind the each key
                np.save(fpath, value)
        else:
            raise NotImplementedError

        return feats_dict
