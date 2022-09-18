import sys
from pathlib import Path

import pytest

sys.path.insert(0, '.')  # nopep8

from models.i3d.extract_i3d import ExtractI3D as Extractor
from tests.utils import base_test_script

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name

# True when run for the first time, then must be False
TO_MAKE_REF = False

# we separate them here because pwc is not supported from `torch_zoo` conda environement
# This checks for the python path and adjusts testing parameters
signature = 'device, video_paths, streams, flow_type, stack_size, step_size, extraction_fps, to_make_ref'
if '/pwc/' in sys.executable:
    test_params = [
        ('cuda:0', './sample/v_GGSY1Qvo990.mp4', None, 'pwc', None, None, None, TO_MAKE_REF),
        ('cuda:0', './sample/v_GGSY1Qvo990.mp4', None, 'pwc', 24, 24, 25, TO_MAKE_REF),
        ('cuda:0', './sample/v_GGSY1Qvo990.mp4', None, 'pwc', 24, 12, 15, TO_MAKE_REF),
        # ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'rgb', 'pwc', None, None, None, TO_MAKE_REF), # this one does not work on import vs cli
        ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'flow', 'pwc', None, None, None, TO_MAKE_REF),
    ]
else:
    test_params = [
        ('cuda:0', './sample/v_GGSY1Qvo990.mp4', None, 'raft', None, None, None, TO_MAKE_REF),
        ('cuda:0', './sample/v_GGSY1Qvo990.mp4', None, 'raft', 24, 24, 25, TO_MAKE_REF),
        ('cuda:0', './sample/v_GGSY1Qvo990.mp4', None, 'raft', 24, 12, 15, TO_MAKE_REF),
        ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'rgb', 'raft', None, None, None, TO_MAKE_REF),
        ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'flow', 'raft', None, None, None, TO_MAKE_REF),
    ]


@pytest.mark.parametrize(signature, test_params)
def test(device, video_paths, streams, flow_type, stack_size, step_size, extraction_fps, to_make_ref):
    # get config
    patch_kwargs = dict(
        device=device,
        video_paths=video_paths,
        streams=streams,
        flow_type=flow_type,
        stack_size=stack_size,
        step_size=step_size,
        extraction_fps=extraction_fps
    )
    base_test_script(FEATURE_TYPE, Extractor, to_make_ref, **patch_kwargs)
