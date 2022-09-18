import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import torch
from omegaconf import OmegaConf

from utils.utils import (build_cfg_path, load_numpy, load_pickle, make_path, sanity_check)


def md5sum(path: str):
    '''The same as `md5sum <file>` in Linux'''
    with open(path, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def make_ref_path(feature_type, file_key, **patch_kwargs):
    filename = ''
    for k, v in patch_kwargs.items():
        if k == 'device':
            continue
        if k == 'video_paths':
            v = Path(v).stem
        if k == 'model_name':
            v = v.replace('/', '_')
        filename += f'{v}_'
    filename += f'{file_key}.pt'
    ref_path = Path('./tests') / feature_type / 'reference' / filename
    return ref_path


def make_ref(args, video_path: Path, data, save_path):
    assert not save_path.exists(), 'Do you make another ref????'
    save_path.parent.mkdir(exist_ok=True)
    to_save = {
        'args': args,
        'video_path': video_path,
        'video_path_md5': md5sum(video_path),
        'data': data,
    }
    torch.save(to_save, save_path)
    print('Saved to', save_path)


def get_config(feature_type, **patch_kwargs):
    config = OmegaConf.load(build_cfg_path(feature_type))
    for k, v in patch_kwargs.items():
        setattr(config, k, v)
    sanity_check(config)
    return config


def get_import_api_feats(extractor, video_paths):
    feat_out_import = extractor.extract(video_paths)
    return feat_out_import


def get_cmd_api_feats(feature_type: str, file_keys: List[str], **patch_kwargs):
    with tempfile.TemporaryDirectory(suffix='_todel_video_features') as output_root:
        # to reduce the number of code lines, these two dicts are created
        action2loadfn = {'save_numpy': load_numpy, 'save_pickle': load_pickle}
        action2ext = {'save_numpy': '.npy', 'save_pickle': '.pkl'}
        feat_out_cmd = {k: dict() for k in action2loadfn.keys()}

        # we are going to test both: `save_numpy` and `save_pickle`
        for on_extraction in action2loadfn.keys():
            # make a cmd (the quotation of numeric arguments might lead to unwanted fails :/)
            cmd = f'{sys.executable} main.py'
            cmd += f' feature_type={feature_type}'
            for k, v in patch_kwargs.items():
                # skips if None, and 0, empty list or dict() but if False does not skip
                cmd += f' {k}={v}' if (isinstance(v, bool) or v) else ''
            cmd += f' on_extraction={on_extraction}'
            cmd += f' output_path={output_root}'
            # call the cmd
            subprocess.call(cmd.split())
            print(cmd)

            # some feature_types do not have model names. Plus, CLIP have `/` in the name
            model_name = patch_kwargs.get('model_name', '').replace('/', '_')
            # TODO: for search: `output_path`
            # TODO: fix this as it now hard-coded by during extration it is in `sanity_check()`
            output_root_load = Path(output_root) / feature_type / model_name

            # read from the saved file
            for key in file_keys:
                load_path = Path(
                    make_path(output_root_load, patch_kwargs['video_paths'], key, action2ext[on_extraction]))
                assert load_path.exists(), (load_path, output_root_load)
                feat_out_cmd[on_extraction][key] = action2loadfn[on_extraction](str(load_path))

    return feat_out_cmd


# TODO: replace it with numpy's or torch's all close
def all_close(a, b, tol=1e-6) -> bool:
    '''Determines if tensors/values `a` and `b` are close to each other given a tolerance'''
    return abs(a - b).sum() < tol


def base_test_script(feature_type: str, Extractor, to_make_ref: bool, **patch_kwargs):
    args = get_config(feature_type, **patch_kwargs)
    # get the model
    extractor = Extractor(args)
    output_feat_keys = extractor.output_feat_keys
    # calculate features: CLI API and import API
    feat_out_cmd = get_cmd_api_feats(feature_type, output_feat_keys, **patch_kwargs)
    feat_out_import = get_import_api_feats(extractor, patch_kwargs['video_paths'])
    print(feat_out_cmd, feat_out_import)
    # tests
    for k in output_feat_keys:
        # TODO: reuse something like all_close function instead
        # compare features saved by pickle and numpy.
        assert all_close(feat_out_cmd['save_pickle'][k], feat_out_cmd['save_numpy'][k])
        # compare cmd API and import API.
        assert all_close(feat_out_cmd['save_numpy'][k], feat_out_import[k])
        # Assuming if it passes these tests, only `feat_out` will be used
        feat_out = feat_out_import[k]
        # load/make the reference (make ref will make the reference file which will be used to compare with)
        ref_path = make_ref_path(feature_type, k, **patch_kwargs)
        if to_make_ref:
            make_ref(args, patch_kwargs['video_paths'], feat_out, ref_path)
        feat_ref = torch.load(ref_path)['data']
        print(k)
        print(feat_out - feat_ref)
        # compare shapes
        assert feat_out.shape == feat_ref.shape
        # compare values
        assert all_close(feat_out, feat_ref)
