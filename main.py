from omegaconf import OmegaConf
from tqdm import tqdm
import time
from pathlib import Path
from functools import partial

from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check


def get_extractor_and_params(args_cli):
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
        from models.i3d.extract_i3d import ExtractI3D as Extractor
    elif args.feature_type == 'r21d':
        from models.r21d.extract_r21d import ExtractR21D as Extractor
    elif args.feature_type == 's3d':
        from models.s3d.extract_s3d import ExtractS3D as Extractor
    elif args.feature_type == 'vggish':
        from models.vggish.extract_vggish import ExtractVGGish as Extractor
    elif args.feature_type == 'resnet':
        from models.resnet.extract_resnet import ExtractResNet as Extractor
    elif args.feature_type == 'raft':
        from models.raft.extract_raft import ExtractRAFT as Extractor
    elif args.feature_type == 'pwc':
        from models.pwc.extract_pwc import ExtractPWC as Extractor
    elif args.feature_type == 'clip':
        from models.clip.extract_clip import ExtractCLIP as Extractor
    else:
        raise NotImplementedError(f'Extractor {args.feature_type} is not implemented.')

    extractor = Extractor(args)

    # unifies whatever a user specified as paths into a list of paths
    video_paths = form_list_from_user_input(args.video_paths, args.file_with_video_paths, to_shuffle=True)
    print(f'The number of specified videos: {len(video_paths)}')

    return extractor, video_paths, args


def main(extractor, args, video_paths):
    for video_path in tqdm(video_paths):
        extractor._extract(video_path)  # note the `_` in the method name

if __name__ == '__main__':
    args_cli = OmegaConf.from_cli()
    _extractor, _vp, args = get_extractor_and_params(args_cli)

    if not args.slurm.submit:
        main(_extractor, args, _vp)
    else:
        try:
            import submitit
        except ImportError:
            print("submitit not installed, cannot submit jobs")
            exit(1)

        # function we want to submit
        subfunc = partial(main, _extractor, args)

        # split the list of videos into chunks
        assert args.slurm.num_arrays <= len(_vp)
        path_lists = [_vp[i::args.slurm.num_arrays] for i in range(args.slurm.num_arrays)]

        # submit jobs
        log_folder_path = Path(args.slurm.log_folder_path)
        executor = submitit.AutoExecutor(
            folder=log_folder_path / f"%j_{args.slurm.name}_{args.feature_type}"
        )
        print("Submitting jobs to the cluster")
        executor.update_parameters(
            slurm_job_name=args.slurm.name,
            mem_gb=args.slurm.mem_gb,
            gpus_per_node=1,   # each task takes one gpu
            tasks_per_node=1,  # one task per GPU
            nodes=1,           # one node per task
            cpus_per_task=args.slurm.cpus_per_task,
            timeout_min=int(args.slurm.timeout_hrs * 60),
            slurm_partition=args.slurm.partition,
            slurm_signal_delay_s=120,
            slurm_array_parallelism=args.slurm.max_arrays,
            # These are hardcoded for VGG cluster...
            # slurm_exclude="gnodec1,gnodec2,gnodec3,gnodec4,gnodeb1",
        )
        jobs = executor.map_array(subfunc, path_lists)

        if args.slurm.monitor:
            # wait and check how many have finished
            num_finished = sum(job.done() for job in jobs)
            # wait and check how many have finished
            while num_finished < len(jobs):
                time.sleep(360)
                num_finished = sum(job.done() for job in jobs)
                print(
                    "Feature extraction:\n \t total: ",
                    len(jobs),
                    "\n \t finished: ",
                    num_finished,
                    end='\r'
                )
