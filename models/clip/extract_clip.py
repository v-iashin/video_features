import os
from typing import Dict, Tuple, Union, Callable
import pathlib

from PIL import Image
import cv2
import numpy as np
import torch
from tqdm import tqdm
from utils.utils import (action_on_extraction, form_list_from_user_input,
                         reencode_video_with_diff_fps)

# try:
#     import clip
# except ModuleNotFoundError:
#     raise ModuleNotFoundError("CLIP module not found\n RUN pip install git+https://github.com/openai/CLIP.git")
# import traceback
# import models.clip.clip_src as clip
from . import clip_src as clip


class ExtractCLIP(torch.nn.Module):

    def __init__(self, args):
        super(ExtractCLIP, self).__init__()
        self.feature_type = args.feature_type
        self.model_name = args.model_name
        self.path_list = form_list_from_user_input(args)
        self.batch_size = args.batch_size
        self.extraction_fps = args.extraction_fps
        # not used, create an issue if you would like to save the frames
        self.keep_tmp_files = args.keep_tmp_files
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
        model, preprocess = self.load_model(device)

        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                feats_dict = self.extract(device, model, preprocess, self.path_list[idx])
                action_on_extraction(feats_dict, self.path_list[idx], self.output_path, self.on_extraction)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # prints only the last line of an error. Use `traceback.print_exc()` for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]} with error (↑). Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, device: torch.device, model: torch.nn.Module, preprocess: Callable,
                video_path: Union[str, None] = None) -> Dict[str, np.ndarray]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            model {torch.nn.Module}
            preprocess {Callable} -- function to preprocess images

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model"-fashion (default: {None})

        Returns:
            Dict[str, np.ndarray]: 'features_nme', 'fps', 'timestamps_ms'
        '''
        def _run_on_a_batch(vid_feats, batch, model, device):
            batch = torch.cat(batch).to(device)

            with torch.no_grad():
                batch_feats = model.encode_image(batch)
                vid_feats.extend(batch_feats.tolist())
                # show predicitons on imagenet dataset (might be useful for debugging)
                # if self.show_pred:
                #     logits = classifier(batch_feats)
                #     show_predictions_on_dataset(logits, 'imagenet')

        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # read a video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamps_ms = []
        batch = []
        vid_feats = []

        # sometimes when the target fps is 1 or 2, the first frame of the reencoded video is missing
        # and cap.read returns None but the rest of the frames are ok. timestep is 0.0 for the 2nd frame in
        # this case
        first_frame = True
        while cap.isOpened():
            frame_exists, rgb = cap.read()

            if first_frame:
                first_frame = False
                if frame_exists is False:
                    continue

            if frame_exists:
                timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                # prepare data (first -- transform, then -- unsqueeze)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = preprocess(Image.fromarray(rgb))
                rgb = rgb.unsqueeze(0)
                batch.append(rgb)
                # when batch is formed to inference
                if len(batch) == self.batch_size:
                    _run_on_a_batch(vid_feats, batch, model, device)
                    # clean up the batch list
                    batch = []
            else:
                # if the last batch was smaller than the batch size
                if len(batch) != 0:
                    _run_on_a_batch(vid_feats, batch, model, device)
                cap.release()
                break

        # removes the video with different fps if it was created to preserve disk space
        if (self.extraction_fps is not None) and (not self.keep_tmp_files):
            os.remove(video_path)

        features_with_meta = {
            self.feature_type: np.array(vid_feats),
            'fps': np.array(fps),
            'timestamps_ms': np.array(timestamps_ms)
        }

        return features_with_meta

    def load_model(self, device: torch.device) -> Tuple[torch.nn.Module, Callable]:
        '''Defines the models, loads checkpoints, sends them to the device.

        Args:
            device (torch.device): The device

        Raises:
            NotImplementedError: if flow type is not implemented.

        Returns:
            Tuple[torch.nn.Module, Callable]: the model and the transform function
        '''
        if self.model_name in clip.available_models():
            model, preprocess = clip.load(self.model_name, device=device)
        elif self.model_name == 'custom':
            # Reserved methods for using custom weights
            # *There is a bug in original repo when loading not-jit weights,
            # *and ignore it for now.
            model_path = os.path.join(pathlib.Path(__file__).parent, 'checkpoints', 'CLIP-custom.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"{model_path}")
            model, preprocess = clip.load(model_path, device=device)
        else:
            raise NotImplementedError(f"Model {self.model_name} not found")
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
        # if self.feature_type == 'CLIP-ViT-B-32':
        #     model, preprocess = clip.load("ViT-B/32", device=device)
        # elif self.feature_type == 'CLIP-ViT-B-16':
        #     model, preprocess = clip.load("ViT-B/16", device=device)
        # elif self.feature_type == 'CLIP-RN50x16':
        #     model, preprocess = clip.load("RN50x16", device=device)
        # elif self.feature_type == 'CLIP-RN50x4':
        #     model, preprocess = clip.load("RN50x4", device=device)
        # elif self.feature_type == 'CLIP-RN101':
        #     model, preprocess = clip.load("RN101", device=device)
        # elif self.feature_type == 'CLIP-RN50':
        #     model, preprocess = clip.load("RN50", device=device)
        # elif self.feature_type == 'CLIP-custom':
        #     # Reserved methods for using custom weights
        #     # *There is a bug in original repo when loading not-jit weights,
        #     # *and ignore it for now.
        #     model_path = os.path.join(pathlib.Path(__file__).parent, 'checkpoints', 'CLIP-custom.pth')
        #     if not os.path.exists(model_path):
        #         raise FileNotFoundError(f"{model_path}")
        #     model, preprocess = clip.load(model_path, device=device)
        # else:
        #     raise NotImplementedError(self.feature_type)
        model.eval()
        return model, preprocess
