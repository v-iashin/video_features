import os
from typing import Dict, List, Union

import cv2
import numpy as np
import torch
import torchvision
import utils.flow_viz as flow_viz
from models._base.base_extractor import BaseExtractor
from models.raft.raft_src.raft import RAFT, InputPadder
from models.transforms import (PILToTensor, ResizeImproved, ToFloat,
                               ToTensorWithoutScaling)
from utils.utils import dp_state_to_normal
from utils.io import VideoLoader


class BaseOpticalFlowExtractor(BaseExtractor):
    '''Common things for all frame-wise extractors (such as RAFT and PWC).'''

    def __init__(self,
                 # BaseExtractor arguments
                 feature_type: str,
                 on_extraction: str,
                 tmp_path: str,
                 output_path: str,
                 keep_tmp_files: bool,
                 device: str,
                 # This class
                 ckpt_path: str,
                 batch_size: int,
                 resize_to_smaller_edge: bool,
                 side_size: Union[None, int],
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
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size
        self.resize_to_smaller_edge = resize_to_smaller_edge
        self.side_size = side_size
        if self.side_size is not None:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                ResizeImproved(self.side_size, self.resize_to_smaller_edge),
                PILToTensor(),
                ToFloat(),
            ])
        else:
            self.transforms = torchvision.transforms.Compose([ToTensorWithoutScaling()])
        self.extraction_fps = extraction_fps  # use `None` to skip reencoding and keep the original video fps
        self.extraction_total = extraction_total
        self.output_feat_keys = [self.feature_type, 'fps', 'timestamps_ms']
        self.show_pred = show_pred
        self.name2module = self.load_model()

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
            batch_size=self.batch_size + 1,  # two frames generate one flow. Add one to generate batch_size flows.
            fps=self.extraction_fps,
            tmp_path=self.tmp_path,
            keep_tmp=self.keep_tmp_files,
            transform=lambda x: self.transforms(x).unsqueeze(0),
            overlap=1
        )
        vid_feats = []
        timestamps_ms_list, timestamps_ms = [], []
        for batch, ts, idx in video:
            # batch = torch.stack(batch, dim=0)
            padder = InputPadder(batch[0].shape) if self.feature_type == 'raft' else None
            batch_feats = self.run_on_a_batch(batch, padder)
            vid_feats.extend(batch_feats.tolist())
            timestamps_ms_list.append(ts)
        for i, ts in enumerate(timestamps_ms_list):
            timestamps_ms.extend(ts if i == 0 else ts[1:])

        features_with_meta = {
            self.feature_type: np.array(vid_feats),
            'fps': np.array(video.fps),
            'timestamps_ms': np.array(timestamps_ms)
        }

        return features_with_meta

    def run_on_a_batch(self, batch: List[torch.Tensor], padder=None) -> torch.Tensor:
        model = self.name2module['model']
        batch = torch.cat(batch).to(self.device)
        if padder is not None:
            batch = padder.pad(batch)

        batch_feats = model(batch[:-1], batch[1:])
        # maybe un-padding only before saving because np.concat will not work if the img is unpadded
        if padder is not None:
            batch_feats = padder.unpad(batch_feats)
        self.maybe_show_pred(batch_feats, batch)
        return batch_feats

    def load_model(self) -> torch.nn.Module:
        '''Defines the models, loads checkpoints, sends them to the device.

        Returns:
            torch.nn.Module: flow extraction module.
        '''
        if self.feature_type == 'raft':
            model = RAFT()
        elif self.feature_type == 'pwc':
            # imported here to avoid environment mismatch when run under `torch_zoo`
            from models.pwc.pwc_src.pwc_net import PWCNet
            model = PWCNet()
        else:
            raise NotImplementedError(f'Flow model {self.feature_type} is not implemented')
        state_dict = torch.load(self.ckpt_path, map_location='cpu')
        state_dict = dp_state_to_normal(state_dict)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return {'model': model}

    def maybe_show_pred(self, batch_feats: torch.Tensor, batch: torch.Tensor):
        '''Shows the resulting flow frames and a corrsponding RGB frame (the 1st of the two) in a cv2 window.

        Args:
            batch_feats (torch.Tensor): the output of the model
            batch (torch.Tensor): the stack of rgb inputs
            device (torch.device, optional): _description_. Defaults to None.
        '''
        if self.show_pred:
            for idx, flow in enumerate(batch_feats):
                img = batch[idx].permute(1, 2, 0).cpu().numpy()
                flow = flow.permute(1, 2, 0).cpu().numpy()
                flow = flow_viz.flow_to_image(flow)
                img_flow = np.concatenate([img, flow], axis=0)
                cv2.imshow('Press any key to see the next frame...', img_flow[:, :, [2, 1, 0]] / 255.0)
                cv2.waitKey()
