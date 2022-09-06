import os
from typing import Dict, Union, List

import cv2
import numpy as np
import torch
from models._base.base_extractor import BaseExtractor
from utils.utils import reencode_video_with_diff_fps


class BaseFrameWiseExtractor(BaseExtractor):
    """Common things for all frame-wise extractors (such as ResNet or CLIP).
    However, optical flow has another parent class: OpticalFlowExtractor"""

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
        self.output_feat_keys = [self.feature_type, 'fps', 'timestamps_ms']
        self.show_pred = show_pred

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """Extracts features for a given video path.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: 'features_name', 'fps', 'timestamps_ms'
        """
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
                rgb = self.transforms(rgb)
                rgb = rgb.unsqueeze(0)
                batch.append(rgb)
                # when batch is formed to inference
                if len(batch) == self.batch_size:
                    batch_feats = self.run_on_a_batch(batch)
                    vid_feats.extend(batch_feats.tolist())
                    # clean up the batch list
                    batch = []
            else:
                # if the last batch was smaller than the batch size
                if len(batch) != 0:
                    batch_feats = self.run_on_a_batch(batch)
                    vid_feats.extend(batch_feats.tolist())
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

    def run_on_a_batch(self, batch: List[torch.Tensor]) -> torch.Tensor:
        model = self.name2module['model']
        batch = torch.cat(batch).to(self.device)
        batch_feats = model(batch)
        self.maybe_show_pred(batch_feats)
        return batch_feats
