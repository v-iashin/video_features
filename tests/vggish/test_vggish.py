import pickle
import numpy as np
import subprocess
import sys
import tempfile
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, '.')  # nopep8

from models.vggish.extract_vggish import ExtractVGGish as Extractor
from tests.utils import make_ref
from utils.utils import build_cfg_path, on_after_sanity_check, sanity_check

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name


# True when run for the first time, then must be False
TO_MAKE_REF = False
@pytest.mark.parametrize('device, video_path, to_make_ref', [
    ('cuda', './sample/v_GGSY1Qvo990.mp4', TO_MAKE_REF),
])
def test(device, video_path, to_make_ref):
    # calculate features with the 'import' API
    # output
    args = OmegaConf.load(build_cfg_path(FEATURE_TYPE))
    args.video_paths = video_path
    sanity_check(args)
    on_after_sanity_check(args)
    # init the extractor
    extractor = Extractor(args)
    model = extractor.load_model(device)
    feat_out_import = extractor.extract(device, model, video_path)

    # calculate features with the 'cmd' API
    with tempfile.TemporaryDirectory(suffix='_todel_video_features') as output_root:
        ways_to_save = ['save_numpy', 'save_pickle']
        feat_out_cmd = {k: dict() for k in ways_to_save}
        # we are going to test both: `save_numpy` and `save_pickle`
        for on_extraction in ways_to_save:
            # make a cmd (the quotation of arguments might lead to unwanted fails :/)
            cmd = f'{sys.executable} main.py'
            cmd += f' feature_type="{FEATURE_TYPE}"'
            cmd += ' device_ids=0' if device == 'cuda' else ' cpu="true"'
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
                load_path_stub = Path(output_root) / FEATURE_TYPE / f'{Path(video_path).stem}_{key}'
                if on_extraction == 'save_numpy':
                    assert load_path_stub.with_suffix('.npy').exists(), (load_path_stub, output_root)
                    feat = np.load(load_path_stub.with_suffix('.npy'))
                elif on_extraction == 'save_pickle':
                    assert load_path_stub.with_suffix('.pkl').exists()
                    feat = pickle.load(open(load_path_stub.with_suffix('.pkl'), 'rb'))
                feat_out_cmd[on_extraction][key] = feat

    feat_out = feat_out_import[FEATURE_TYPE]
    # load/make the reference (make ref will make the reference file which will be used to compare with)
    filename = f'{device}_{Path(video_path).stem}.pt'
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
