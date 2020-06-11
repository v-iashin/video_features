import os
import subprocess

def form_list_from_user_input(args):
    if args.file_with_video_paths is not None:
        with open(args.file_with_video_paths) as rfile:
            # remove carriage return
            path_list = [line.replace('\n', '') for line in rfile.readlines()]
            # remove empty lines
            path_list = [path for path in path_list if len(path) > 0]
    else:
        path_list = args.video_paths

    return path_list


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path


def fix_tensorflow_gpu_allocation(argparse_args):
    '''tf somehow makes it impossible to specify a GPU index in the code, hence we use the env variable.
    To address this, we will assign the user-defined cuda ids to environment variable CUDA_VISIBLE_DEVICES.

    For example: if user specifies --device_ids 1 3 5 we will assign 1,3,5 to CUDA_VISIBLE_DEVICES environment
                 variable and reassign args.device_ids with [0, 1, 2] which are indices to the list of
                 user specified ids [1, 3, 5].

    Args:
        argparse_args (args): user-defined arguments from argparse
    '''
    # argparse_args.device_ids are ints which cannot be joined with ','.join()
    device_ids = [str(index) for index in argparse_args.device_ids]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_ids)
    # [1, 3, 5] -> [0, 1, 2]
    argparse_args.device_ids = list(range(len(argparse_args.device_ids)))
