
import omegaconf
from models._base.base_flow_extractor import BaseOpticalFlowExtractor

# defined as a constant here, because i3d imports it
DATASET_to_PWC_CKPT_PATHS = {
    'sintel': './models/pwc/checkpoints/pwc_net_sintel.pt'
}


class ExtractPWC(BaseOpticalFlowExtractor):

    def __init__(self, args: omegaconf.DictConfig) -> None:
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
            ckpt_path=DATASET_to_PWC_CKPT_PATHS['sintel'],
            batch_size=args.batch_size,
            resize_to_smaller_edge=args.resize_to_smaller_edge,
            side_size=args.side_size,
            extraction_fps=args.extraction_fps,
            extraction_total=args.extraction_total,
            show_pred=args.show_pred,
        )
