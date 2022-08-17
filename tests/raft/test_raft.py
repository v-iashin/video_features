import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, '.')  # nopep8

from models.raft.extract_raft import ExtractRAFT as Extractor
from tests.utils import make_ref
from utils.utils import build_cfg_path

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name


# True when run for the first time, then must be False
TO_MAKE_REF = False
@pytest.mark.parametrize('device, video_path, finetuned_on, batch_size, side_size, resize_to_smaller_edge, extraction_fps, to_make_ref', [
    # ('cuda', './sample/v_GGSY1Qvo990.mp4', 'sintel', 1, None, True, None, TO_MAKE_REF), # 500MB+
    # ('cuda', './sample/v_GGSY1Qvo990.mp4', 'kitti', 1, None, True, None, TO_MAKE_REF), # 500MB+
    # ('cuda', './sample/v_GGSY1Qvo990.mp4', 'sintel', 16, None, True, None, TO_MAKE_REF), # 500MB+
    # ('cuda', './sample/v_GGSY1Qvo990.mp4', 'sintel', 1, 256, False, None, TO_MAKE_REF), # 500MB+
    ('cuda', './sample/v_GGSY1Qvo990.mp4', 'sintel', 1, None, True, 1, TO_MAKE_REF), # 26M
    ('cuda', './sample/v_GGSY1Qvo990.mp4', 'kitti', 1, 256, True, 1, TO_MAKE_REF), # 26M
    ('cuda', './sample/v_GGSY1Qvo990.mp4', 'sintel', 16, 256, False, 1, TO_MAKE_REF), # 17M
    ('cuda', './sample/v_GGSY1Qvo990.mp4', 'sintel', 1, 256, False, 1, TO_MAKE_REF), # 17M
])
def test(device, video_path, finetuned_on, batch_size, side_size, resize_to_smaller_edge, extraction_fps, to_make_ref):
    # output
    args = OmegaConf.load(build_cfg_path(FEATURE_TYPE))
    args.video_paths = video_path
    args.finetuned_on = finetuned_on
    args.batch_size = batch_size
    args.side_size = side_size
    args.resize_to_smaller_edge = resize_to_smaller_edge
    args.extraction_fps = extraction_fps
    extractor = Extractor(args)
    model = extractor.load_model(device)
    features_out = extractor.extract(device, model, video_path)
    features_out = features_out[FEATURE_TYPE]
    # load/make the reference (make ref will make the reference file which will be used to compare with)
    filename = f'{device}_{Path(video_path).stem}_{finetuned_on}_{batch_size}_{side_size}_{resize_to_smaller_edge}_{extraction_fps}.pt'
    ref_path = Path('./tests') / FEATURE_TYPE / 'reference' / filename
    if to_make_ref:
        make_ref(args, video_path, features_out, ref_path)
    features_ref = torch.load(ref_path)['data']
    # tests
    print(features_out)
    print(features_out.shape)
    assert features_out.shape == features_ref.shape
    assert (features_out - features_ref).sum() < 1e-6
