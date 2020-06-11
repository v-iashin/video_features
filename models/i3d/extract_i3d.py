# import argparse
import os
import shutil
# import sys
from typing import Dict, Union

import numpy as np
import torch
from tqdm import tqdm
# import traceback

# sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models.i3d.flow_src.pwc_net import PWCNet
from models.i3d.i3d_src.i3d_feats import i3d_features
from models.i3d.i3d_src.i3d_net import I3D_RGB_FLOW
from models.i3d.utils.utils import extract_frames_from_video, form_iter_list
from utils.utils import form_list_from_user_input

class ExtractI3D(torch.nn.Module):

    def __init__(self, args):
        super(ExtractI3D, self).__init__()
        self.path_list = form_list_from_user_input(args)
        self.pwc_path = args.pwc_path
        self.i3d_rgb_path = args.i3d_rgb_path
        self.i3d_flow_path = args.i3d_flow_path
        self.min_side_size = args.min_side_size
        self.extraction_fps = args.extraction_fps
        self.step_size = args.step_size
        self.stack_size = args.stack_size
        self.show_kinetics_pred = args.show_kinetics_pred
        self.kinetics_class_labels = args.kinetics_class_labels
        self.keep_frames = args.keep_frames
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
        
        pwc_model = PWCNet(self.pwc_path).to(device)
        i3d_model = I3D_RGB_FLOW(self.i3d_rgb_path, self.i3d_flow_path).to(device)

        for idx in indices:
            # when error occurs might fail silently when run from torch data parallel
            try:
                self.extract(device, pwc_model, i3d_model, idx)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # traceback.print_exc()  # for the whole traceback
                print(e)
                print(f'Extraction failed at: {self.path_list[idx]}. Continuing extraction')

            # update tqdm progress bar
            self.progress.update()

    def extract(self, device: torch.device, pwc_model: torch.nn.Module, i3d_model: torch.nn.Module, 
                idx: int, video_path: Union[str, None] = None
                ) -> Dict[str, Union[torch.nn.Module, str]]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            pwc_model {torch.nn.Module}
            i3d_model {torch.nn.Module}
            idx {int} -- index to self.video_paths

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as 
                                             "path -> i3d features"-fashion (default: {None})

        Returns:
            Dict[str, Union[torch.nn.Module, str]] -- dict with i3d features and their type
        '''
        if video_path is None:
            video_path = self.path_list[idx]

        frames_dir = extract_frames_from_video(video_path, self.extraction_fps, 
                                               self.min_side_size, self.tmp_path)
        # sorted list of frame paths
        frame_paths = [os.path.join(frames_dir, fname) for fname in sorted(os.listdir(frames_dir))]
        # T+1, since T flow frames require T+1 rgb frames
        frame_paths = form_iter_list(frame_paths, self.step_size, self.stack_size+1)
        # extract features
        i3d_feats = i3d_features(
            frame_paths, self.stack_size, device, pwc_model, i3d_model,
            self.show_kinetics_pred, self.kinetics_class_labels
        )

        # removes the folder with extracted frames to preserve disk space
        if not self.keep_frames:
            shutil.rmtree(frames_dir)

        # What to do once features are extracted.
        if self.on_extraction == 'print':
            print(i3d_feats)
        elif self.on_extraction == 'save_numpy':
            # make dir if doesn't exist
            os.makedirs(self.output_path, exist_ok=True)
            # extract file name and change the extention
            filename_rgb = os.path.split(video_path)[-1].replace('.mp4', '_rgb.npy')
            filename_flow = os.path.split(video_path)[-1].replace('.mp4', '_flow.npy')
            # construct the paths to save the features
            feature_rgb_path = os.path.join(self.output_path, filename_rgb)
            feature_flow_path = os.path.join(self.output_path, filename_flow)
            # save features
            np.save(feature_rgb_path, i3d_feats['rgb'].cpu())
            np.save(feature_flow_path, i3d_feats['flow'].cpu())
        else:
            raise NotImplementedError

        return i3d_feats


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Extract I3D Features')
#     parser.add_argument('--video_paths', nargs='+', help='space-separated paths to videos')
#     parser.add_argument('--file_with_video_paths', help='.txt file where each line is a path')
#     parser.add_argument('--tmp_path', default='../tmp'
#                         help='folder to store the extracted frames before the extraction')
#     parser.add_argument('--output_path', default='./output', help='where to store results if saved')
#     parser.add_argument('--pwc_path', default='./checkpoints/pwc_net.pt')
#     parser.add_argument('--i3d_rgb_path', default='./checkpoints/i3d_rgb.pt')
#     parser.add_argument('--i3d_flow_path', default='./checkpoints/i3d_flow.pt')
#     parser.add_argument('--min_side_size', type=int, default=256, help='min(HEIGHT, WIDTH)')
#     parser.add_argument('--extraction_fps', type=int, help='Do not specify for original video fps')
#     parser.add_argument('--stack_size', type=int, default=64, help='Feature time span in fps')
#     parser.add_argument('--step_size', type=int, default=64, help='Feature step size in fps')
#     parser.add_argument('--device_id', type=int, default=0, help='Device to run the extraction on')
#     parser.add_argument(
#         '--show_kinetics_pred', dest='show_kinetics_pred', action='store_true', default=False,
#         help='to show the predictions of the i3d model into kinetics classes for each feature'
#     )
#     parser.add_argument('--kinetics_class_labels', default='./checkpoints/label_map.txt')
#     parser.add_argument(
#         '--keep_frames', dest='keep_frames', action='store_true', default=False,
#         help='to keep frames after feature extraction'
#     )
#     parser.add_argument(
#         '--on_extraction', default='print', choices=['print', 'save_numpy'],
#         help='what to do once the stack is extracted'
#     )

#     args = parser.parse_args()

#     if torch.cuda.is_available():
#         video_paths = form_list_from_user_input(args)
#         # create a list of indices [0, video_paths_length]. ExtractI3D will use them and their
#         # device in the forward pass
#         indices = torch.arange(len(video_paths), device=torch.device(args.device_id))
#         # init the module and call forward pass
#         i3d_extractor = ExtractI3D(args)
#         i3d_extractor(indices)
#     else:
#         raise Exception('Cannot detect a cuda device. Sorry, it is implemented for GPU-only.')
