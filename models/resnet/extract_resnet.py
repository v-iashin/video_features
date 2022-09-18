from typing import Dict
import omegaconf

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from models._base.base_framewise_extractor import BaseFrameWiseExtractor
from utils.utils import show_predictions_on_dataset


class ExtractResNet(BaseFrameWiseExtractor):

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
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.name2module = self.load_model()


    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.

        Raises:
            NotImplementedError: if a model is not implemented.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        try:
            model = getattr(models, self.model_name)
        except AttributeError:
            raise NotImplementedError(f'Model {self.model_name} not found.')

        model = model(pretrained=True)
        model = model.to(self.device)
        model.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        class_head = model.fc
        model.fc = torch.nn.Identity()
        return {
            'model': model,
            'class_head': class_head,
        }

    def maybe_show_pred(self, feats: torch.Tensor):
        if self.show_pred:
            logits = self.name2module['class_head'](feats)
            show_predictions_on_dataset(logits, 'imagenet')
