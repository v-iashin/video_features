
import omegaconf
from typing import Dict, List
import torch
from PIL import Image
from torchvision.transforms import Compose
from video_features.models._base.base_framewise_extractor import BaseFrameWiseExtractor
from video_features.utils.utils import show_predictions_on_dataset

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError:
    raise ImportError("This features require timm library to be installed.")

class ExtractTIMM(BaseFrameWiseExtractor):

    def __init__(self, args: omegaconf.DictConfig) -> None:
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
            model_name=args.model_name,
            batch_size=args.batch_size,
            extraction_fps=args.extraction_fps,
            extraction_total=args.extraction_total,
            show_pred=args.show_pred,
        )

        # transform must be implemented in _create_model
        self.transforms = None
        self.name2module = self.load_model()

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints and related transforms,
        sends them to the device.

        Raises:
            NotImplementedError: if a model is not implemented.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        model = timm.create_model(self.model_name, pretrained=True)

        # transforms
        self.transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.transforms = Compose([lambda np_array: Image.fromarray(np_array), self.transforms])
        print(self.transforms)

        model.to(self.device)
        model.eval()

        # remove the classifier after getting it
        class_head = model.get_classifier()
        model.reset_classifier(0)

        # to be used in `run_on_a_batch` to determine the how to show predictions
        self.hf_arch = model.default_cfg['architecture']
        self.hf_tag = model.default_cfg.get('tag', '')

        return {'model': model, 'class_head': class_head, }

    def run_on_a_batch(self, batch: List) -> torch.Tensor:
        """This is a hack for timm models to output features.
        Ideally, you want to use model_spec to define behaviour at forward pass in
        the config file.
        """
        model = self.name2module['model']
        batch = torch.cat(batch).to(self.device)
        batch_feats = model(batch)
        self.maybe_show_pred(batch_feats)
        return batch_feats

    def maybe_show_pred(self, feats: torch.Tensor):
        if self.show_pred:
            logits = self.name2module['class_head'](feats)
            # NOTE: these hardcoded ends assume that the end of the tag corresponds to the last training dset
            if self.hf_tag.endswith(('in1k', 'in1k_288', 'in1k_320', 'in1k_384', 'in1k_475', 'in1k_512',)):
                show_predictions_on_dataset(logits, 'imagenet1k')
            elif self.hf_tag.endswith(('in21k', 'in21k_288', 'in21k_320', 'in21k_384', 'in21k_475',
                                       'in21k_512',
                                       'in22k', 'in22k_288', 'in22k_320', 'in22k_384', 'in22k_475',
                                       'in22k_512',)):
                show_predictions_on_dataset(logits, 'imagenet21k')
            else:
                print(f'No show_pred for {self.hf_arch} with tag {self.hf_tag}; use `show_pred=False`')
