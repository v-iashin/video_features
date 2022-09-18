import os
from typing import Dict, Union, List

import cv2
import numpy as np
import torch
from models._base.base_extractor import BaseExtractor
from utils.io import VideoLoader


class BaseFrameWiseExtractor(BaseExtractor):
    '''Common things for all frame-wise extractors (such as ResNet or CLIP).
    However, optical flow has another parent class: OpticalFlowExtractor'''

    def __init__(self,
                 # BaseExtractor arguments
                 feature_type: str,
                 on_extraction: str,
                 tmp_path: str,
                 output_path: str,
                 keep_tmp_files: bool,
                 device: str,
                 # This class
                 model_name: str,
                 batch_size: int,
                 extraction_fps: Union[None, int],
                 extraction_total: Union[None, int],
                 show_pred: bool,
                 ) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=feature_type,
            on_extraction=on_extraction,
            tmp_path=tmp_path,
            output_path=output_path,
            keep_tmp_files=keep_tmp_files,
            device=device,
        )
        # (Re-)Define arguments for this class
        self.model_name = model_name
        self.batch_size = batch_size
        self.extraction_fps = extraction_fps  # use `None` to skip reencoding and keep the original video fps
        self.extraction_total = extraction_total
        self.output_feat_keys = [self.feature_type, 'fps', 'timestamps_ms']
        self.show_pred = show_pred

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        '''Extracts features for a given video path.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: 'features_name', 'fps', 'timestamps_ms'
        '''

        video = VideoLoader(
            video_path,
            batch_size=self.batch_size,
            fps=self.extraction_fps,
            total=self.extraction_total,
            tmp_path=self.tmp_path,
            keep_tmp=self.keep_tmp_files,
            transform=lambda x: self.transforms(x).unsqueeze(0)
        )
        vid_feats = []
        timestamps_ms = []
        for batch, timestamp_ms, idx in video:
            # batch = torch.stack(batch, dim=0)
            batch_feats = self.run_on_a_batch(batch)
            vid_feats.extend(batch_feats.tolist())
            timestamps_ms.extend(timestamp_ms)

        features_with_meta = {
            self.feature_type: np.array(vid_feats),
            'fps': np.array(video.fps),
            'timestamps_ms': np.array(timestamps_ms)
        }

        return features_with_meta

    def run_on_a_batch(self, batch: List[torch.Tensor]) -> torch.Tensor:
        model = self.name2module['model']
        batch = torch.cat(batch).to(self.device)
        batch_feats = model(batch)
        self.maybe_show_pred(batch_feats)
        return batch_feats
