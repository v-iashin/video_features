import os
import pathlib
from typing import Dict, Union

import numpy as np
import torch
from tqdm import tqdm
import traceback

from utils.utils import form_list_from_user_input, extract_wav_from_mp4, action_on_extraction
from models.vggish.vggish_src.vggish_slim import VGGish


class ExtractVGGish(torch.nn.Module):

    def __init__(self, args):
        super(ExtractVGGish, self).__init__()
        self.feature_type = args.feature_type
        self.path_list = form_list_from_user_input(args)
        self.keep_tmp_files = args.keep_tmp_files
        self.on_extraction = args.on_extraction
        self.tmp_path = os.path.join(args.tmp_path, self.feature_type)
        self.output_path = os.path.join(args.output_path, self.feature_type)
        self.progress = tqdm(total=len(self.path_list))
        if args.show_pred:
            raise NotImplementedError

    def forward(self, indices: torch.LongTensor):
        '''
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        '''
        device = indices.device
        model = self.load_model(device)

        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                feats_dict = self.extract(device, model, self.path_list[idx])
                action_on_extraction(feats_dict, self.path_list[idx], self.output_path, self.on_extraction)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # prints only the last line of an error. Use `traceback.print_exc()` for the whole traceback
                traceback.print_exc()
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]} with error (↑). Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self,
                device: torch.device,
                model: torch.nn.Module,
                video_path: Union[str, None] = None) -> Dict[str, np.ndarray]:
        '''The extraction call. Made to clean the forward call a bit.

        Args:
            device (torch.device): the device
            model (torch.nn.Module): the model
            video_path (Union[str, None], optional): Path to a video. Defaults to None.

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model"-fashion (default: {None})

        Returns:
            Dict[str, np.ndarray]: extracted VGGish features
        '''
        file_ext = pathlib.Path(video_path).suffix

        if file_ext == '.mp4':
            # extract audio files from .mp4
            audio_wav_path, audio_aac_path = extract_wav_from_mp4(video_path, self.tmp_path)
        elif file_ext == '.wav':
            audio_wav_path = video_path
            audio_aac_path = None
        else:
            raise NotImplementedError

        with torch.no_grad():
            vggish_stack = model(audio_wav_path, device).cpu().numpy()

        # removes the folder with audio files created during the process
        if not self.keep_tmp_files:
            if video_path.endswith('.mp4'):
                os.remove(audio_wav_path)
                os.remove(audio_aac_path)

        feats_dict = {self.feature_type: vggish_stack}

        return feats_dict

    def load_model(self, device: torch.device) -> torch.nn.Module:
        '''Defines the models, loads checkpoints, sends them to the device.

        Args:
            device (torch.device)

        Returns:
            torch.nn.Module: the model
        '''
        model = VGGish()
        model = model.to(device)
        model.eval()
        return model
