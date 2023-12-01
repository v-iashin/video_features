
import omegaconf
from typing import Dict, List
import torch
from PIL import Image
from torchvision.transforms import Compose
from models._base.base_framewise_extractor import BaseFrameWiseExtractor

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError:
    raise ImportError("This features require timm library to be installed.")

class ExtractFrames(BaseFrameWiseExtractor):
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
        
        # TODO: this is a hack, ideally you want to use model spec to define
        # behaviour at forward pass. ATM I'm just overriding
        # to spit out features `run_on_a_batch_function`
        # example of how it could be done:
        # model = timm.create_model(self.model_spec['model'], num_classes=0, global_pool='')
        model = timm.create_model(self.model_name)
        self.transforms =  create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.transforms = Compose([lambda np_array: Image.fromarray(np_array), self.transforms])
        print(self.transforms)
        model.to(self.device)
        model.eval()
        return {"model": model}

    def run_on_a_batch(self, batch: List) -> torch.Tensor:
        """This is a hack for timm models to output features.
        Ideally, you want to use model_spec to define behaviour at forward pass in
        the config file.
        """
        model = self.name2module['model']
        batch = torch.cat(batch).to(self.device)
        batch_feats = model.forward_features(batch)
        # FIXME: ignoring likeaboss
        # self.maybe_show_pred(batch_feats)
        return batch_feats