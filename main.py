# we import unused `numpy` before `torch` because if run from `subprocess.run()`
# it fails with
# `Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.`
# (see https://github.com/pytorch/pytorch/issues/37377)
import numpy
import torch
from omegaconf import OmegaConf

from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check

def parallel_feature_extraction(args):
    '''Distributes the feature extraction in embarasingly-parallel fashion. Specifically,
    it divides the dataset (list of video paths) among all specified devices evenly and extract features.'''

    if args.feature_type == 'i3d':
        from models.i3d.extract_i3d import ExtractI3D  # defined here to avoid import errors
        extractor = ExtractI3D(args)
    elif args.feature_type == 'r21d':
        from models.r21d.extract_r21d import ExtractR21D  # defined here to avoid import errors
        extractor = ExtractR21D(args)
    elif args.feature_type == 'vggish':
        from models.vggish.extract_vggish import ExtractVGGish  # defined here to avoid import errors
        extractor = ExtractVGGish(args)
    elif args.feature_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        from models.resnet.extract_resnet import ExtractResNet
        extractor = ExtractResNet(args)
    elif args.feature_type == 'raft':
        from models.raft.extract_raft import ExtractRAFT
        extractor = ExtractRAFT(args)
    elif args.feature_type == 'pwc':
        from models.pwc.extract_pwc import ExtractPWC
        extractor = ExtractPWC(args)
    else:
        raise NotADirectoryError

    # the indices correspond to the positions of the target videos in
    # the video_paths list. They are required here because
    # scatter module inputs only tensors but there is no such torch tensor
    # that would be suitable for strings (video_paths). Also, the
    # input have the method '.device' which allows us to access the
    # current device in the extractor.
    video_paths = form_list_from_user_input(args)
    indices = torch.arange(len(video_paths))
    replicas = torch.nn.parallel.replicate(extractor, args.device_ids[:len(indices)])
    inputs = torch.nn.parallel.scatter(indices, args.device_ids[:len(indices)])
    torch.nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)
    # closing the tqdm progress bar to avoid some unexpected errors due to multi-threading
    extractor.progress.close()


if __name__ == "__main__":
    cfg_cli = OmegaConf.from_cli()
    print(cfg_cli)
    cfg_yml = OmegaConf.load(build_cfg_path(cfg_cli.feature_type))
    # the latter arguments are prioritized
    cfg = OmegaConf.merge(cfg_yml, cfg_cli)
    # OmegaConf.set_readonly(cfg, True)
    print(OmegaConf.to_yaml(cfg))
    # some printing
    if cfg.on_extraction in ['save_numpy', 'save_pickle']:
        print(f'Saving features to {cfg.output_path}')
    if cfg.keep_tmp_files:
        print(f'Keeping temp files in {cfg.tmp_path}')

    sanity_check(cfg)
    parallel_feature_extraction(cfg)
