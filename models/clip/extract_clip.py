import pathlib
from typing import Dict
import omegaconf

import torch
from models._base.base_framewise_extractor import BaseFrameWiseExtractor
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from utils.utils import KINETICS_CLASS_PATH, show_predictions_on_dataset

from . import clip_src as clip


class ExtractCLIP(BaseFrameWiseExtractor):

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
        self.transforms = 'For CLIP, it is easier to define in .load_model method because we need input size'
        if self.show_pred:
            pred_texts = args.get('pred_texts', None)
            # if the user didn't specify custom text descriptions, do zero-shot on Kinetics 400
            if pred_texts is None:
                self.pred_texts = [f'a photo of {x.strip()}' for x in open(KINETICS_CLASS_PATH)]
            else:
                self.pred_texts = list(pred_texts)
            # .long() is required because torch.nn.Embedding allows only Longs for pytorch 1.7.1
            self.pred_texts_tok = clip.tokenize(self.pred_texts).long()
        self.name2module = self.load_model()

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.
        For CLIP, it also sets the appropriate transforms

        Raises:
            NotImplementedError: if a model is not implemented.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        if self.model_name in clip.available_models():
            model_path = self.model_name
        elif self.model_name == 'custom':
            # Reserved methods for using custom weights
            # *There is a bug in original repo when loading not-jit weights,
            # *and ignore it for now.
            model_path = pathlib.Path(__file__).parent / 'checkpoints' / 'CLIP-custom.pth'
            if not model_path.exists():
                raise FileNotFoundError(f'{model_path}')
        else:
            raise NotImplementedError(f'Model {self.model_name} not found')

        model, _ = clip.load(str(model_path), device=self.device)
        model.eval()

        # defining transforms
        # doing it here instead of __init__ because it is cleaner to access model input size from here
        input_size = model.visual.input_resolution
        self.transforms = Compose([
            lambda np_array: Image.fromarray(np_array),
            Resize(input_size, interpolation=Image.BICUBIC),
            CenterCrop(input_size),
            lambda image: image.convert('RGB'),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        return {
            'model': model.encode_image,
            'model.encode_text': model.encode_text,
            'model.logit_scale': model.logit_scale,
        }

    def maybe_show_pred(self, visual_feats: torch.Tensor):
        # for each batch we will compute text representation: it is a bit redundant but it only
        # creates a problem during `show_pred`, i.e. debugging. It is not a big deal
        if self.show_pred:
            # to(device) is called here (instead of __init__) because device is defined in .extract()
            text_feats = self.name2module['model.encode_text'](self.pred_texts_tok.to(self.device))

            # visual_feats:T, 512  text_feats:N, 512
            visual_feats = visual_feats.to(device=self.device, dtype=torch.double)
            text_feats = text_feats.to(dtype=torch.double)

            # normalized features
            visual_feats = visual_feats / visual_feats.norm(dim=1, keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.name2module['model.logit_scale'].exp().to(dtype=visual_feats.dtype)
            logits = logit_scale * visual_feats @ text_feats.t()
            logits = logits.cpu()

            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # T, N

            show_predictions_on_dataset(logits, self.pred_texts)
