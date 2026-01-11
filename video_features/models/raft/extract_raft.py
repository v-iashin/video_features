
import omegaconf
from video_features.models._base.base_flow_extractor import BaseOpticalFlowExtractor
from video_features.utils.utils import get_package_file_path

# defined as a constant here, because i3d imports it
# Use package-relative paths that work when installed
DATASET_to_RAFT_CKPT_PATHS = {
    'sintel': str(get_package_file_path('models/raft/checkpoints/raft-sintel.pth')),
    'kitti': str(get_package_file_path('models/raft/checkpoints/raft-kitti.pth')),
}


class ExtractRAFT(BaseOpticalFlowExtractor):

    def __init__(self, args: omegaconf.DictConfig) -> None:
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
            ckpt_path=DATASET_to_RAFT_CKPT_PATHS[args.finetuned_on],
            batch_size=args.batch_size,
            resize_to_smaller_edge=args.resize_to_smaller_edge,
            side_size=args.side_size,
            extraction_fps=args.extraction_fps,
            extraction_total=args.extraction_total,
            show_pred=args.show_pred,
        )
