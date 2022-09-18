import sys
from pathlib import Path

import pytest

sys.path.insert(0, '.')  # nopep8

from models.r21d.extract_r21d import ExtractR21D as Extractor
from tests.utils import base_test_script

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name

# True when run for the first time, then must be False
TO_MAKE_REF = False

signature = 'device, video_paths, model_name, stack_size, step_size, extraction_fps, to_make_ref'
test_params = [
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'r2plus1d_18_16_kinetics', None, None, None, TO_MAKE_REF),
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'r2plus1d_34_32_ig65m_ft_kinetics', None, None, None, TO_MAKE_REF),
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'r2plus1d_34_8_ig65m_ft_kinetics', None, None, None, TO_MAKE_REF),
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'r2plus1d_34_8_ig65m_ft_kinetics', None, None, 1, TO_MAKE_REF),
]


@pytest.mark.parametrize(signature, test_params)
def test(device, video_paths, model_name, stack_size, step_size, extraction_fps, to_make_ref):
    # get config
    patch_kwargs = dict(
        device=device,
        video_paths=video_paths,
        model_name=model_name,
        stack_size=stack_size,
        step_size=step_size,
        extraction_fps=extraction_fps
    )
    base_test_script(FEATURE_TYPE, Extractor, to_make_ref, **patch_kwargs)
