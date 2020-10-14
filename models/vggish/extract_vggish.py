import os
import pathlib
from typing import Dict, Union

import numpy as np
import torch
from tqdm import tqdm
# import traceback

from utils.utils import form_list_from_user_input, extract_wav_from_mp4, action_on_extraction
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

VGGISH_MODEL_PATH = './models/vggish/checkpoints/vggish_model.ckpt'
VGGISH_PCA_PATH = './models/vggish/checkpoints/vggish_pca_params.npz'

class ExtractVGGish(torch.nn.Module):

    def __init__(self, args):
        super(ExtractVGGish, self).__init__()
        self.feature_type = args.feature_type
        self.path_list = form_list_from_user_input(args)
        self.vggish_model_path = VGGISH_MODEL_PATH
        self.vggish_pca_path = VGGISH_PCA_PATH
        self.keep_tmp_files = args.keep_tmp_files
        self.on_extraction = args.on_extraction
        self.tmp_path = os.path.join(args.tmp_path, self.feature_type)
        self.output_path = os.path.join(args.output_path, self.feature_type)
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
                    feats_dict = self.extract(sess, self.path_list[idx])
                    action_on_extraction(feats_dict, self.path_list[idx], self.output_path, self.on_extraction)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    # traceback.print_exc()  # for the whole traceback
                    print(e)
                    print(f'Extraction failed at: {self.path_list[idx]}. Continuing extraction')

                # update tqdm progress bar
                self.progress.update()

    def extract(self, tf_session, video_path: Union[str, None] = None) -> Dict[str, np.ndarray]:
        '''The extraction call. Made to clean the forward call a bit.

        Args:
            tf_session (tensorflow.python.client.session.Session): tf session
            video_path (Union[str, None], optional): . Defaults to None.

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

        # extract features (credits: tensorflow models)
        examples_batch = vggish_input.wavfile_to_examples(audio_wav_path)
        features_tensor = tf_session.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = tf_session.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        [vggish_stack] = tf_session.run([embedding_tensor], feed_dict={features_tensor: examples_batch})

        # removes the folder with audio files created during the process
        if not self.keep_tmp_files:
            if video_path.endswith('.mp4'):
                os.remove(audio_wav_path)
                os.remove(audio_aac_path)

        feats_dict = {
            self.feature_type: vggish_stack
        }

        return feats_dict
