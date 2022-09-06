import os
import pathlib
from typing import Dict

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
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
        )
        # (Re-)Define arguments for this class
        if args.show_pred:
            raise NotImplementedError
        self.output_feat_keys = [self.feature_type]
        self.name2module = self.load_model()

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """Extracts features for a given video path.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
        """
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
            vggish_stack = self.name2module['model'](audio_wav_path, self.device).cpu().numpy()

        # removes the folder with audio files created during the process
        if not self.keep_tmp_files:
            if video_path.endswith('.mp4'):
                os.remove(audio_wav_path)
                os.remove(audio_aac_path)

        feats_dict = {self.feature_type: vggish_stack}

        return feats_dict

    def load_model(self) -> torch.nn.Module:
        """Defines the models, loads checkpoints, sends them to the device.


        Returns:
            {Dict[str, torch.nn.Module]}: model-agnostic dict holding modules for extraction
        """
        model = VGGish()
        model = model.to(self.device)
        model.eval()
        return {'model': model}
