import os
import pathlib
from typing import Dict, Union

import numpy as np
import torch
from models._base.base_extractor import BaseExtractor
from models.vggish.vggish_src.vggish_slim import VGGish
from utils.utils import extract_wav_from_mp4


class ExtractVGGish(BaseExtractor):

    def __init__(self, args) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=args.feature_type,
            video_paths=args.video_paths,
            file_with_video_paths=args.file_with_video_paths,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
        )
        # (Re-)Define arguments for this class
        if args.show_pred:
            raise NotImplementedError
        self.output_feat_keys = [self.feature_type]

    @torch.no_grad()
    def extract(
        self,
        device: torch.device,
        name2module: Dict[str, torch.nn.Module],
        video_path: Union[str, None] = None
    ) -> Dict[str, np.ndarray]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            name2module {Dict[str, torch.nn.Module]}: model-agnostic dict holding modules for extraction

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model"-fashion (default: {None})

        Returns:
            Dict[str, np.ndarray]: 'features_name', 'fps', 'timestamps_ms'
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
            vggish_stack = name2module['model'](audio_wav_path, device).cpu().numpy()

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
            {Dict[str, torch.nn.Module]}: model-agnostic dict holding modules for extraction
        '''
        model = VGGish()
        model = model.to(device)
        model.eval()
        return {'model': model}
