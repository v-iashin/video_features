import torch
import argparse

from utils.utils import form_list_from_user_input, fix_tensorflow_gpu_allocation


def parallel_feature_extraction(args):
    '''Distributes the feature extraction in a embarasingly-parallel fashion. Specifically,
    it divides the dataset (list of video paths) among all specified devices evenly.'''

    if args.feature_type == 'i3d':
        from models.i3d.extract_i3d import ExtractI3D  # defined here to avoid import errors
        extractor = ExtractI3D(args)
    elif args.feature_type == 'vggish':
        from models.vggish.extract_vggish import ExtractVGGish  # defined here to avoid import errors
        fix_tensorflow_gpu_allocation(args)
        extractor = ExtractVGGish(args)

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
    # Main args
    parser.add_argument('--feature_type', required=True, choices=['i3d', 'vggish'])
    parser.add_argument('--video_paths', nargs='+', help='space-separated paths to videos')
    parser.add_argument('--file_with_video_paths', help='.txt file where each line is a path')
    parser.add_argument('--device_ids', type=int, nargs='+', help='space-separated device ids')
    parser.add_argument('--tmp_path', default='./tmp',
                        help='folder to store the extracted frames before the extraction')
    parser.add_argument('--keep_frames', dest='keep_frames', action='store_true', default=False,
                        help='to keep frames after feature extraction')
    parser.add_argument('--on_extraction', default='print', choices=['print', 'save_numpy'],
                        help='what to do once the stack is extracted')
    parser.add_argument('--output_path', default='./output', help='where to store results if saved')
    # I3D options
    parser.add_argument('--pwc_path', default='./models/i3d/checkpoints/pwc_net.pt')
    parser.add_argument('--i3d_rgb_path', default='./models/i3d/checkpoints/i3d_rgb.pt')
    parser.add_argument('--i3d_flow_path', default='./models/i3d/checkpoints/i3d_flow.pt')
    parser.add_argument('--min_side_size', type=int, default=256, help='min(HEIGHT, WIDTH)')
    parser.add_argument('--extraction_fps', type=int, help='Do not specify for original video fps')
    parser.add_argument('--stack_size', type=int, default=64, help='Feature time span in fps')
    parser.add_argument('--step_size', type=int, default=64, help='Feature step size in fps')
    parser.add_argument(
        '--show_kinetics_pred', dest='show_kinetics_pred', action='store_true', default=False,
        help='to show the predictions of the i3d model into kinetics classes for each feature'
    )
    parser.add_argument('--kinetics_class_labels', default='./checkpoints/label_map.txt')
    # VGGish options
    parser.add_argument('--vggish_model_path', default='./models/vggish/checkpoints/vggish_model.ckpt')
    parser.add_argument('--vggish_pca_path', default='./models/vggish/checkpoints/vggish_pca_params.npz')

    args = parser.parse_args()

    # some printing
    if args.on_extraction == 'save_numpy':
        print(f'Saving features to {args.output_path}')
    if args.keep_frames:
        print(f'Keeping temp files in {args.tmp_path}')

    parallel_feature_extraction(args)
