import sys
from pathlib import Path

import pytest

sys.path.insert(0, '.')  # nopep8

from video_features.models.timm import ExtractTIMM as Extractor
from tests.utils import base_test_script

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name

# True when run for the first time, then must be False
TO_MAKE_REF = False

signature = 'device, video_paths, model_name, batch_size, extraction_fps, to_make_ref'
test_params = [
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'vit_base_patch16_224.dino', 1, 1, TO_MAKE_REF),
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'coat_tiny.in1k', 1, 1, TO_MAKE_REF),
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'hf-hub:nateraw/resnet50-oxford-iiit-pet', 4, 2, TO_MAKE_REF),
    ('cuda:0', './sample/v_GGSY1Qvo990.mp4', 'mobilenetv3_small_050', 1, None, TO_MAKE_REF),
]


@pytest.mark.parametrize(signature, test_params)
def test(device, video_paths, model_name, batch_size, extraction_fps, to_make_ref):
    # get config
    patch_kwargs = dict(
        device=device,
        video_paths=video_paths,
        model_name=model_name,
        batch_size=batch_size,
        extraction_fps=extraction_fps
    )
    base_test_script(FEATURE_TYPE, Extractor, to_make_ref, **patch_kwargs)
