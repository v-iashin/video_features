import pickle
import numpy as np
import tempfile
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, '.')  # nopep8

from models.r21d.extract_r21d import ExtractR21D as Extractor
from tests.utils import make_ref
from utils.utils import build_cfg_path, on_after_sanity_check, sanity_check

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name


# True when run for the first time, then must be False
TO_MAKE_REF = False
@pytest.mark.parametrize('device, video_path, model_name, stack_size, step_size, extraction_fps, to_make_ref', [
    ('cuda', './sample/v_GGSY1Qvo990.mp4', 'r2plus1d_18_16_kinetics', None, None, None, TO_MAKE_REF),
    ('cuda', './sample/v_GGSY1Qvo990.mp4', 'r2plus1d_34_32_ig65m_ft_kinetics', None, None, None, TO_MAKE_REF),
    ('cuda', './sample/v_GGSY1Qvo990.mp4', 'r2plus1d_34_8_ig65m_ft_kinetics', None, None, None, TO_MAKE_REF),
    ('cuda', './sample/v_GGSY1Qvo990.mp4', 'r2plus1d_34_8_ig65m_ft_kinetics', None, None, 1, TO_MAKE_REF),
])
def test(device, video_path, model_name, stack_size, step_size, extraction_fps, to_make_ref):
    # calculate features with the 'import' API
    args = OmegaConf.load(build_cfg_path(FEATURE_TYPE))
    # patch the config
    args.video_paths = video_path
    args.model_name = model_name
    args.stack_size = stack_size
    args.step_size = step_size
    args.extraction_fps = extraction_fps
    sanity_check(args)
    on_after_sanity_check(args)
    # init the extractor
    extractor = Extractor(args)
    model, class_head = extractor.load_model(device)
    # extract features
    feat_out_import = extractor.extract(device, model, class_head, video_path)

    # calculate features with the 'cmd' API
    with tempfile.TemporaryDirectory(suffix='_todel_video_features') as output_root:
        ways_to_save = ['save_numpy', 'save_pickle']
        feat_out_cmd = {k: dict() for k in ways_to_save}
        # we are going to test both: `save_numpy` and `save_pickle`
        for on_extraction in ways_to_save:
            # make a cmd (the quotation of arguments might lead to unwanted fails :/)
            cmd = f'{sys.executable} main.py'
            cmd += f' feature_type="{FEATURE_TYPE}"'
            cmd += f' model_name="{model_name}"' if model_name else ''
            cmd += ' device_ids=0' if device == 'cuda' else ' cpu="true"'
            cmd += f' stack_size={stack_size}' if stack_size else ''
            cmd += f' step_size={step_size}' if step_size else ''
            cmd += f' extraction_fps="{extraction_fps}"' if extraction_fps else ''
            cmd += f' on_extraction="{on_extraction}"' if on_extraction else ''
            cmd += f' video_paths={video_path}'
            cmd += f' output_path="{output_root}"'
            # damn, it is ugly
            # call the cmd
            subprocess.call(cmd.split())
            # read from the saved file
            for key in feat_out_import.keys():
                # for search: `output_path`
                # TODO: fix this as it now hard-coded by during extration it is in `on_after_sanity_check`
                load_path_stub = Path(output_root) / FEATURE_TYPE / model_name / f'{Path(video_path).stem}_{key}'
                if on_extraction == 'save_numpy':
                    assert load_path_stub.with_suffix('.npy').exists(), (load_path_stub, output_root)
                    feat = np.load(load_path_stub.with_suffix('.npy'))
                elif on_extraction == 'save_pickle':
                    assert load_path_stub.with_suffix('.pkl').exists()
                    feat = pickle.load(open(load_path_stub.with_suffix('.pkl'), 'rb'))
                feat_out_cmd[on_extraction][key] = feat

    # compare features saved by pickle and numpy
    assert (feat_out_cmd['save_pickle'][FEATURE_TYPE] - feat_out_cmd['save_numpy'][FEATURE_TYPE]).sum() < 1e-6
    # compare cmd API and import API. Assuming if it passes these tests, only `feat_out` will be used
    assert (feat_out_cmd['save_numpy'][FEATURE_TYPE] - feat_out_import[FEATURE_TYPE]).sum() < 1e-6

    # Assuming if it passes the previous tests, I will use only the feats from 'import' API
    feat_out = feat_out_import[FEATURE_TYPE]
    # load/make the reference (make ref will make the reference file which will be used to compare with)
    filename = f'{device}_{Path(video_path).stem}_{model_name}_{stack_size}_{step_size}_{extraction_fps}.pt'
    ref_path = Path('./tests') / FEATURE_TYPE / 'reference' / filename
    if to_make_ref:
        make_ref(args, video_path, feat_out, ref_path)
    feat_ref = torch.load(ref_path)['data']
    # tests
    print(feat_out)
    print(feat_out.shape)
    print(feat_ref)
    print(feat_ref.shape)
    assert feat_out.shape == feat_ref.shape
    assert (feat_out - feat_ref).sum() < 1e-6
