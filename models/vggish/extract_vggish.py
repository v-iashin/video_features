import os
import shutil
from typing import Dict, Union

import numpy as np
import torch
from tqdm import tqdm
# import traceback

from utils.utils import form_list_from_user_input
from models.vggish.utils.utils import extract_wav_from_mp4
from models.vggish.vggish_src import (vggish_input, vggish_params,
                                      vggish_postprocess, vggish_slim)

import tensorflow as tf

# turn off a ton of warnings produced by TF
import logging
if type(tf.contrib) != type(tf):
    tf.contrib._warning = None
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# tensorflow config
cfg = tf.ConfigProto(allow_soft_placement=True)
cfg.gpu_options.allow_growth = True

class ExtractVGGish(torch.nn.Module):

    def __init__(self, args):
        super(ExtractVGGish, self).__init__()
        self.path_list = form_list_from_user_input(args)
        self.vggish_model_path = args.vggish_model_path
        self.vggish_pca_path = args.vggish_pca_path
        self.keep_audio_files = args.keep_frames  # naming problem, yes. :TODO
        self.on_extraction = args.on_extraction
        self.tmp_path = args.tmp_path
        self.output_path = args.output_path
        self.progress = tqdm(total=len(self.path_list))

    def forward(self, indices: torch.LongTensor):
        '''
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        '''
        device_id = indices.device.index

        # Define the model in inference mode, load the model, and
        # locate input and output tensors.
        # (credits: tensorflow models)
        with tf.Graph().as_default(), tf.Session(config=cfg) as sess, tf.device(f'/device:GPU:{device_id}'):
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, self.vggish_model_path)
            pproc = vggish_postprocess.Postprocessor(self.vggish_pca_path)

            # iterate over the list of videos
            for idx in indices:
                # when error occurs might fail silently when run from torch data parallel
                try:
                    self.extract(idx, sess)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    # traceback.print_exc()  # for the whole traceback
                    print(e)
                    print(f'Extraction failed at: {self.path_list[idx]}. Continuing extraction')

                # update tqdm progress bar
                self.progress.update()

    def extract(self, idx: int, tf_session, video_path: Union[str, None] = None) -> np.ndarray:
        '''The extraction call. Made to clean the forward call a bit.

        Args:
            idx (int): index to self.path_list
            tf_session (tensorflow.python.client.session.Session): tf session
            video_path (Union[str, None], optional): . Defaults to None.

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as 
                                             "path -> i3d features"-fashion (default: {None})

        Returns:
            np.ndarray: extracted VGGish features
        '''
        # if video_path is not specified, take one from the self.path_list
        if video_path is None:
            video_path = self.path_list[idx]

        # extract audio files from .mp4
        audio_wav_path, audio_aac_path = extract_wav_from_mp4(video_path, self.tmp_path)

        # extract features (credits: tensorflow models)
        examples_batch = vggish_input.wavfile_to_examples(audio_wav_path)
        features_tensor = tf_session.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = tf_session.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        [vggish_stack] = tf_session.run([embedding_tensor], feed_dict={features_tensor: examples_batch})

        # removes the folder with extracted frames to preserve disk space
        if not self.keep_audio_files:
            os.remove(audio_wav_path)
            os.remove(audio_aac_path)

        # What to do once features are extracted.
        if self.on_extraction == 'print':
            print(vggish_stack)
            # print(vggish_stack.sum())
        elif self.on_extraction == 'save_numpy':
            # make dir if doesn't exist
            os.makedirs(self.output_path, exist_ok=True)
            # extract file name and change the extention
            filename = os.path.split(video_path)[-1].replace('.mp4', '_vggish.npy')
            # construct the paths to save the features
            feature_path = os.path.join(self.output_path, filename)
            # save features
            np.save(feature_path, vggish_stack)
        else:
            raise NotImplementedError
        
        return vggish_stack

