import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, '.')  # nopep8

from models.i3d.extract_i3d import ExtractI3D as Extractor
from tests.utils import make_ref
from utils.utils import build_cfg_path

# a bit ugly but it assumes the features being tested has the same folder name,
# e.g. for r21d: ./tests/r21d/THIS_FILE
# it prevents doing the same tests for different features
THIS_FILE_PATH = __file__
FEATURE_TYPE = Path(THIS_FILE_PATH).parent.name


# True when run for the first time, then must be False
TO_MAKE_REF = False

# we separate them here because pwc is not supported from `torch_zoo` conda environement
# This checks for the python path and adjusts testing parameters
signature = 'device, video_path, streams, flow_type, stack_size, step_size, extraction_fps, to_make_ref'
if '/pwc/' in sys.executable:
    test_params = [
        ('cuda', './sample/v_GGSY1Qvo990.mp4', None, 'pwc', None, None, None, TO_MAKE_REF),
        ('cuda', './sample/v_GGSY1Qvo990.mp4', None, 'pwc', 24, 24, 25, TO_MAKE_REF),
        ('cuda', './sample/v_GGSY1Qvo990.mp4', 'rgb', 'pwc', None, None, None, TO_MAKE_REF),
        ('cuda', './sample/v_GGSY1Qvo990.mp4', 'flow', 'pwc', None, None, None, TO_MAKE_REF),
    ]
else:
    test_params = [
        ('cuda', './sample/v_GGSY1Qvo990.mp4', None, 'raft', None, None, None, TO_MAKE_REF),
        ('cuda', './sample/v_GGSY1Qvo990.mp4', None, 'raft', 24, 24, 25, TO_MAKE_REF),
        ('cuda', './sample/v_GGSY1Qvo990.mp4', 'rgb', 'raft', None, None, None, TO_MAKE_REF),
        ('cuda', './sample/v_GGSY1Qvo990.mp4', 'flow', 'raft', None, None, None, TO_MAKE_REF),
    ]

@pytest.mark.parametrize(signature, test_params)
def test(device, video_path, streams, flow_type, stack_size, step_size, extraction_fps, to_make_ref):
    # output
    args = OmegaConf.load(build_cfg_path(FEATURE_TYPE))
    args.video_paths = video_path
    args.streams = streams
    args.flow_type = flow_type
    args.stack_size = stack_size
    args.step_size = step_size
    args.extraction_fps = extraction_fps
    extractor = Extractor(args)
    model, class_head = extractor.load_model(device)
    features = extractor.extract(device, model, class_head, video_path)
    for k in [key for key in features.keys() if key in ['rgb', 'flow']]:
        features_out = features[k]
        # load/make the reference (make ref will make the reference file which will be used to compare with)
        filename = f'{device}_{Path(video_path).stem}_{streams}_{flow_type}_{stack_size}_{step_size}_{extraction_fps}_{k}.pt'
        ref_path = Path('./tests') / FEATURE_TYPE / 'reference' / filename
        if to_make_ref:
            make_ref(args, video_path, features_out, ref_path)
        features_ref = torch.load(ref_path)['data']
        # tests
        print(k)
        print(features_out)
        print(features_out.shape)
        print(features_ref)
        print(features_ref.shape)
        assert features_out.shape == features_ref.shape
        assert (features_out - features_ref).sum() < 1e-6
