
import omegaconf
from models._base.base_flow_extractor import BaseOpticalFlowExtractor

# defined as a constant here, because i3d imports it
DATASET_to_RAFT_CKPT_PATHS = {
    'sintel': './models/raft/checkpoints/raft-sintel.pth',
    'kitti': './models/raft/checkpoints/raft-kitti.pth',
}


class ExtractRAFT(BaseOpticalFlowExtractor):

    def __init__(self, args: omegaconf.DictConfig) -> None:
        super().__init__(
            feature_type=args.feature_type,
            video_paths=args.video_paths,
            file_with_video_paths=args.file_with_video_paths,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            ckpt_path=DATASET_to_RAFT_CKPT_PATHS[args.finetuned_on],
            batch_size=args.batch_size,
            resize_to_smaller_edge=args.resize_to_smaller_edge,
            side_size=args.side_size,
            extraction_fps=args.extraction_fps,
            show_pred=args.show_pred,
        )
