from typing import Union

import torch
from omegaconf import ListConfig
from tqdm import tqdm
from utils.utils import action_on_extraction, form_list_from_user_input, is_already_exist

# import traceback


class BaseExtractor(torch.nn.Module):
    '''Common things to be inherited by every descendant'''

    def __init__(self,
        feature_type: str,
        video_paths: Union[str, ListConfig],
        file_with_video_paths: str,
        on_extraction: str,
        tmp_path: str,
        output_path: str,
        keep_tmp_files: bool
    ) -> None:
        super().__init__()
        self.feature_type = feature_type
        self.path_list = form_list_from_user_input(video_paths, file_with_video_paths)
        self.on_extraction = on_extraction
        self.tmp_path = tmp_path
        self.output_path = output_path
        self.keep_tmp_files = keep_tmp_files
        self.tqdm_progress = tqdm(total=len(self.path_list))

    def forward(self, indices: torch.LongTensor):
        '''
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        '''
        device = indices.device
        name2module = self.load_model(device)

        for idx in indices:
            video_path = self.path_list[idx]
            # when error occurs might fail silently when run from torch data parallel
            try:
                # self.output_feat_keys must be defined by the child class that inherits `BaseExtractor`
                if not is_already_exist(self.output_path, video_path, self.output_feat_keys, self.on_extraction):
                    feats_dict = self.extract(device, name2module, video_path)
                    action_on_extraction(feats_dict, video_path, self.output_path, self.on_extraction)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                # prints only the last line of an error. Use `traceback.print_exc()` for the whole traceback:
                # traceback.print_exc()
                print(e)
                print(f'Extraction failed at: {video_path} with error (↑). Continuing extraction')

            self.tqdm_progress.update()
