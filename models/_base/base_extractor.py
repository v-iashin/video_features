import os
import traceback
from pathlib import Path
from typing import Dict, Union, List

import numpy as np
from utils.utils import (load_numpy, load_pickle, make_path, write_numpy,
                         write_pickle, write_h5_single_file, load_h5_single_file, video_exists_in_h5, video2group)

class BaseExtractor(object):
    """Common things to be inherited by every descendant"""

    def __init__(self,
                 feature_type: str,
                 on_extraction: str,
                 tmp_path: str,
                 output_path: str,
                 keep_tmp_files: bool,
                 device: str,
                 output_feat_keys: List[str] = None # adds a python list for output feature keys
                 ) -> None:
        self.feature_type = feature_type
        self.on_extraction = on_extraction
        self.tmp_path = tmp_path
        self.output_path = output_path
        self.keep_tmp_files = keep_tmp_files
        self.device = device
        self.output_feat_keys = output_feat_keys if output_feat_keys is not None else []

    def _extract(self, video_path: str):
        """A wrapper around self.extract. It handles exceptions, checks if files already exist and saves
        the extracted files if a user desires.

        Args:
            video_path (str): a video path from which to extract features

        Raises:
            KeyboardInterrupt: when an error occurs, the script will continue with the rest of the videos.
                               If a user wants to kill it, ^C (KB interrupt) should be used.
        """
        # the try and except structure is used to continue extraction even if after an error (a few bad vids)
        try:
            # skips a video path if already exists
            if not self.is_already_exist(video_path):
                # extracts features for a video path (this should be implemented by the child modules)
                feats_dict = self.extract(video_path)
                # either prints or saves to numpy/pickle
                self.action_on_extraction(feats_dict, video_path)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print(f'An error occurred during extraction from: {video_path}:')
            traceback.print_exc()  # prints the error
            print('Continuing...')

    def action_on_extraction(
            self,
            feats_dict: Dict[str, np.ndarray],
            video_path: str,
    ) -> None:
        """What is going to be done with the extracted features.

        Args:
            feats_dict (Dict[str, np.ndarray]): A dict with features and possibly some meta. Key will be used as
                                                suffixes to the saved files if `save_numpy` or `save_pickle` or 'save_h5'
                                                used.
            video_path (str): A path to the video.
        """
        # since the features are enclosed in a dict with another meta information we will iterate on kv
        action2ext = {'save_numpy': '.npy', 'save_pickle': '.pkl', 'save_h5': '.h5'} # added support for HDF5 extension
        action2savefn = {'save_numpy': write_numpy, 'save_pickle': write_pickle, 'save_h5': write_h5_single_file}

        # playing safe: the second check if files already exist and openable before possibly overwritting them
        if self.on_extraction in action2savefn and self.is_already_exist(video_path):
            # it is ok to ignore this warning
            print(f'WARNING: extraction didnt find feature files on the 1st try but did on the 2nd try.')
            return
        
        # HDF5 saving : Saving the extracted features in HDF5 File.
        if self.on_extraction == 'save_h5':
            # Creates a filename specific to the device, e.g., 'video_features_cuda0.h5'
            # replaced ':' with '_' because ':' is not allowed in Windows filenames
            sanitized_device = self.device.replace(":","_")
            # Final filename for device:0 would be video_features_cuda0.h5
            h5_filename = f"video_features_{sanitized_device}.h5"
            h5_path = os.path.join(self.output_path, h5_filename)

            os.makedirs(self.output_path, exist_ok=True)
            for key, value in feats_dict.items():
                if key != 'fps' and len(value) == 0:
                    print(f'Warning: Empty value for {key} @ {h5_path}')
            # save all features in single h5 file
            write_h5_single_file(h5_path, video2group(video_path), feats_dict)
        
        elif self.on_extraction in ['save_numpy', 'save_pickle']:
            for key,value in feats_dict.items():
                os.makedirs(self.output_path, exist_ok=True)
                fpath = make_path(self.output_path, video_path, key, action2ext[self.on_extraction])
                if key != 'fps' and len(value) == 0:
                    print(f'Warning: Empty value for {key} @ {fpath}')
                action2savefn[self.on_extraction](fpath,value)
        
        elif self.on_extraction == 'print':
            for key, value in feats_dict.items():
                if self.on_extraction == 'print':
                    print(key)
                    print(value)
                    print(f'max: {value.max():.8f}; mean: {value.mean():.8f}; min: {value.min():.8f}')
                    print()
        else:
            raise NotImplementedError(f'on_extraction: {self.on_extraction} is not implemented')

    def is_already_exist(
            self,
            video_path: Union[str, Path],
    ) -> bool:
        """Checks if the all feature files already exist, and also checks if IO does not produce any errors.

        Args:
            video_path (Union[str, Path]): the path to a video to extract features from
        """
        # if a user does not want to save any features, we need to extract them. 'False' will continue extraction.
        if self.on_extraction == 'print':
            return False

        action2ext = {'save_numpy': '.npy', 'save_pickle': '.pkl', 'save_h5':'.h5'}
        action2loadfn = {'save_numpy': load_numpy, 'save_pickle': load_pickle, 'save_h5': load_h5_single_file}

        if self.on_extraction == 'save_h5':
            sanitized_device = self.device.replace(':', '_')
            h5_filename = f"video_features_{sanitized_device}.h5"
            h5_path = os.path.join(self.output_path, h5_filename)
            if not os.path.exists(h5_path):
                return False
            return video_exists_in_h5(h5_path, video2group(video_path))

        elif self.on_extraction in ['save_numpy', 'save_pickle']:
            how_many_files_should_exist = len(self.output_feat_keys)
            how_many_files_exist = 0  # it is a counter

            for key in self.output_feat_keys:
                fpath = make_path(self.output_path, video_path, key, action2ext[self.on_extraction])
                if Path(fpath).exists():
                    action2loadfn[self.on_extraction](fpath)
                    how_many_files_exist += 1
                else:
                    return False

        if how_many_files_exist == how_many_files_should_exist:
            print(f'Features for {video_path} already exist in {str(Path(fpath).absolute().parent)}/ - skipping..')
            return True
        else:
            return False
