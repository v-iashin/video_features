'''
    The content of this file is commented to avoid parsing it with pytest test discovery.
    Since PWC optical flow extraction depends on another env
'''
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, '.')  # nopep8

from models.pwc.extract_pwc import ExtractPWC as Extractor
from tests.utils import make_ref
from utils.utils import build_cfg_path

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name


# True when run for the first time, then must be False
TO_MAKE_REF = False
@pytest.mark.parametrize('device, video_path, side_size, extraction_fps, to_make_ref', [
    # ('cuda', './sample/v_GGSY1Qvo990.mp4', None, None, TO_MAKE_REF),  # 559M
    ('cuda', './sample/v_GGSY1Qvo990.mp4', 112, 3, TO_MAKE_REF),  # 18M
    # ('cuda', './sample/v_GGSY1Qvo990.mp4', 384, 5, TO_MAKE_REF),  # 345M
])
def test(device, video_path, side_size, extraction_fps, to_make_ref):
    # output
    args = OmegaConf.load(build_cfg_path(FEATURE_TYPE))
    args.video_paths = video_path
    args.side_size = side_size
    args.extraction_fps = extraction_fps
    extractor = Extractor(args)
    model = extractor.load_model(device)
    features_out = extractor.extract(device, model, video_path)
    features_out = features_out[FEATURE_TYPE]
    # load/make the reference (make ref will make the reference file which will be used to compare with)
    filename = f'{device}_{Path(video_path).stem}_{side_size}_{extraction_fps}.pt'
    ref_path = Path('./tests') / FEATURE_TYPE / 'reference' / filename
    if to_make_ref:
        make_ref(args, video_path, features_out, ref_path)
    features_ref = torch.load(ref_path)['data']
    # tests
    print(features_out)
    print(features_out.shape)
    assert features_out.shape == features_ref.shape
    assert (features_out - features_ref).sum() < 1e-6
