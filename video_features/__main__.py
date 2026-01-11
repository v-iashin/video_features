"""CLI entry point for video-features package."""

from omegaconf import OmegaConf
from tqdm import tqdm

from video_features.utils.utils import build_cfg_path, form_list_from_user_input, sanity_check


def main(args_cli=None):
    """Main entry point for video-features CLI.
    
    Args:
        args_cli: Command-line arguments. If None, will be parsed from sys.argv.
    """
    if args_cli is None:
        args_cli = OmegaConf.from_cli()
    
    # config
    args_yml = OmegaConf.load(build_cfg_path(args_cli.feature_type))
    args = OmegaConf.merge(args_yml, args_cli)  # the latter arguments are prioritized
    # OmegaConf.set_readonly(args, True)
    sanity_check(args)

    # verbosing with the print -- haha (TODO: logging)
    print(OmegaConf.to_yaml(args))
    if args.on_extraction in ['save_numpy', 'save_pickle']:
        print(f'Saving features to {args.output_path}')
    print('Device:', args.device)

    # import are done here to avoid import errors (we have two conda environements)
    if args.feature_type == 'i3d':
        from video_features.models.i3d.extract_i3d import ExtractI3D as Extractor
    elif args.feature_type == 'r21d':
        from video_features.models.r21d.extract_r21d import ExtractR21D as Extractor
    elif args.feature_type == 's3d':
        from video_features.models.s3d.extract_s3d import ExtractS3D as Extractor
    elif args.feature_type == 'vggish':
        from video_features.models.vggish.extract_vggish import ExtractVGGish as Extractor
    elif args.feature_type == 'resnet':
        from video_features.models.resnet.extract_resnet import ExtractResNet as Extractor
    elif args.feature_type == 'raft':
        from video_features.models.raft.extract_raft import ExtractRAFT as Extractor
    elif args.feature_type == 'clip':
        from video_features.models.clip.extract_clip import ExtractCLIP as Extractor
    elif args.feature_type == 'timm':
        from video_features.models.timm.extract_timm import ExtractTIMM as Extractor
    else:
        raise NotImplementedError(f'Extractor {args.feature_type} is not implemented.')

    extractor = Extractor(args)

    # unifies whatever a user specified as paths into a list of paths
    video_paths = form_list_from_user_input(args.video_paths, args.file_with_video_paths, to_shuffle=True)

    print(f'The number of specified videos: {len(video_paths)}')

    for video_path in tqdm(video_paths):
        extractor._extract(video_path)  # note the `_` in the method name

    # yep, it is this simple!


if __name__ == '__main__':
    args_cli = OmegaConf.from_cli()
    main(args_cli)

