# we import unused `numpy` before `torch` because if run from `subprocess.run()`
# it fails with
# `Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.`
# (see https://github.com/pytorch/pytorch/issues/37377)
import numpy
import torch
import argparse

from utils.utils import form_list_from_user_input, fix_tensorflow_gpu_allocation, sanity_check

def parallel_feature_extraction(args):
    '''Distributes the feature extraction in embarasingly-parallel fashion. Specifically,
    it divides the dataset (list of video paths) among all specified devices evenly and extract features.'''

    if args.feature_type == 'i3d':
        from models.i3d.extract_i3d import ExtractI3D  # defined here to avoid import errors
        extractor = ExtractI3D(args)
    elif args.feature_type == 'r21d_rgb':
        from models.r21d.extract_r21d import ExtractR21D  # defined here to avoid import errors
        extractor = ExtractR21D(args)
    elif args.feature_type == 'vggish':
        from models.vggish.extract_vggish import ExtractVGGish  # defined here to avoid import errors
        fix_tensorflow_gpu_allocation(args)
        extractor = ExtractVGGish(args)
    elif args.feature_type in ['resnet50']:
        from models.resnet50.extract_resnet50 import ExtractResNet50
        extractor = ExtractResNet50(args)
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
    parser = argparse.ArgumentParser(description='Extract Features')
    parser.add_argument('--feature_type', required=True,
                        choices=['i3d', 'vggish', 'r21d_rgb', 'resnet50', 'raft', 'pwc'])
    parser.add_argument('--video_paths', nargs='+', help='space-separated paths to videos')
    parser.add_argument('--file_with_video_paths', help='.txt file where each line is a path')
    parser.add_argument('--device_ids', type=int, nargs='+', help='space-separated device ids')
    parser.add_argument('--tmp_path', default='./tmp',
                        help='folder to store the temporary files used for extraction (frames or aud files)')
    parser.add_argument('--keep_tmp_files', dest='keep_tmp_files', action='store_true', default=False,
                        help='to keep temp files after feature extraction. (works only for vggish and i3d)')
    parser.add_argument('--on_extraction', default='print', choices=['print', 'save_numpy'],
                        help='what to do once the stack is extracted')
    parser.add_argument('--output_path', default='./output', help='where to store results if saved')

    parser.add_argument('--extraction_fps', type=int, help='Do not specify for original video fps')
    parser.add_argument('--stack_size', type=int, help='Feature time span in fps')
    parser.add_argument('--step_size', type=int, help='Feature step size in fps')
    parser.add_argument('--streams', nargs='+', choices=['flow', 'rgb'],
                        help='Streams to use for feature extraction. Both used if not specified')
    parser.add_argument('--flow_type', choices=['raft', 'pwc'], default='pwc',
                        help='Flow to use in I3D. PWC is faster while RAFT is more accurate.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batchsize (only frame-wise extractors are supported)')
    parser.add_argument('--resize_to_larger_edge', dest='resize_to_smaller_edge', action='store_false',
                        default=True, help='The larger side will be resized to this number maintaining the'
                        + 'aspect ratio. By default, uses the smaller side (as Resize in torchvision).')
    parser.add_argument('--side_size', type=int,
                        help='If specified, the input images will be resized to this value in RAFT.')
    parser.add_argument(
        '--show_pred', dest='show_pred', action='store_true', default=False,
        help='to show preds of a model, i.e. on a pre-train dataset (imagenet or kinetics) for each feature'
    )

    args = parser.parse_args()

    # some printing
    if args.on_extraction == 'save_numpy':
        print(f'Saving features to {args.output_path}')
    if args.keep_tmp_files:
        print(f'Keeping temp files in {args.tmp_path}')

    sanity_check(args)
    parallel_feature_extraction(args)
