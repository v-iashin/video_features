"""
    The content of this file is commented to avoid parsing it with pytest test discovery.
    Since PWC optical flow extraction depends on another env
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, '.')  # nopep8

from models.pwc.extract_pwc import ExtractPWC as Extractor
from tests.utils import base_test_script

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name

# True when run for the first time, then must be False
TO_MAKE_REF = False

signature = 'device, video_paths, side_size, extraction_fps, to_make_ref'
test_params = [
    # ('cuda:0', './sample/v_GGSY1Qvo990.mp4', None, None, TO_MAKE_REF),  # 559M
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 112, 3, TO_MAKE_REF),  # 18M
    # ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 384, 5, TO_MAKE_REF),  # 345M
]


@pytest.mark.parametrize(signature, test_params)
def test(device, video_paths, side_size, extraction_fps, to_make_ref):
    # get config
    patch_kwargs = dict(
        device=device,
        video_paths=video_paths,
        side_size=side_size,
        extraction_fps=extraction_fps
    )
    base_test_script(FEATURE_TYPE, Extractor, to_make_ref, **patch_kwargs)
