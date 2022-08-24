from omegaconf import OmegaConf

from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check


def main(args_cli):
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
        from models.i3d.extract_i3d import ExtractI3D
        extractor = ExtractI3D(args)
    elif args.feature_type == 'r21d':
        from models.r21d.extract_r21d import ExtractR21D
        extractor = ExtractR21D(args)
    elif args.feature_type == 'vggish':
        from models.vggish.extract_vggish import ExtractVGGish
        extractor = ExtractVGGish(args)
    elif args.feature_type == 'resnet':
        from models.resnet.extract_resnet import ExtractResNet
        extractor = ExtractResNet(args)
    elif args.feature_type == 'raft':
        from models.raft.extract_raft import ExtractRAFT
        extractor = ExtractRAFT(args)
    elif args.feature_type == 'pwc':
        from models.pwc.extract_pwc import ExtractPWC
        extractor = ExtractPWC(args)
    elif args.feature_type == 'clip':
        from models.clip.extract_clip import ExtractCLIP
        extractor = ExtractCLIP(args)
    else:
        raise NotADirectoryError

    # unifies whatever a user specified as paths into a list of paths
    video_paths = form_list_from_user_input(args.video_paths, args.file_with_video_paths)

    print(f'The number of specified videos: {len(video_paths)}')

    for video_path in video_paths:
        extractor._extract(video_path)  # note the `_` in the method name

    # yep, it is this simple!


if __name__ == '__main__':
    args_cli = OmegaConf.from_cli()
    main(args_cli)
