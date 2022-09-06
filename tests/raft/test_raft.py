import sys
from pathlib import Path

import pytest

sys.path.insert(0, '.')  # nopep8

from models.raft.extract_raft import ExtractRAFT as Extractor
from tests.utils import base_test_script

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name

# True when run for the first time, then must be False
TO_MAKE_REF = False

signature = 'device, video_paths, finetuned_on, batch_size, side_size, resize_to_smaller_edge, extraction_fps, to_make_ref'
test_params = [
    # ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'sintel', 1, None, True, None, TO_MAKE_REF), # 500MB+
    # ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'kitti', 1, None, True, None, TO_MAKE_REF), # 500MB+
    # ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'sintel', 16, None, True, None, TO_MAKE_REF), # 500MB+
    # ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'sintel', 1, 256, False, None, TO_MAKE_REF), # 500MB+
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'sintel', 1, None, True, 1, TO_MAKE_REF),  # 26M
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'kitti', 1, 256, True, 1, TO_MAKE_REF),  # 26M
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'sintel', 16, 256, False, 1, TO_MAKE_REF),  # 17M
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'sintel', 1, 256, False, 1, TO_MAKE_REF),  # 17M
]


@pytest.mark.parametrize(signature, test_params)
def test(device, video_paths, finetuned_on, batch_size, side_size, resize_to_smaller_edge, extraction_fps, to_make_ref):
    # get config
    patch_kwargs = dict(
        device=device,
        video_paths=video_paths,
        finetuned_on=finetuned_on,
        batch_size=batch_size,
        side_size=side_size,
        resize_to_smaller_edge=resize_to_smaller_edge,
        extraction_fps=extraction_fps
    )
    base_test_script(FEATURE_TYPE, Extractor, to_make_ref, **patch_kwargs)
