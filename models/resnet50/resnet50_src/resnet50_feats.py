from typing import Any, Dict
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import video

def show_predictions_on_IN(logits: torch.FloatTensor, imagenet_class_path: str):
    '''Prints out predictions using logits extracted from one feature window

    Args:
        logits (torch.FloatTensor): after-classification layer vector
        imagenet_class_path (str): path to labels of Imagenet
    '''
    imagenet_classes = [x.strip() for x in open(imagenet_class_path)]

    # Show predictions
    softmaxes = F.softmax(logits, dim=-1)
    top_val, top_idx = torch.sort(softmaxes, dim=-1, descending=True)

    # print top k classes
    k = 5
    logits_score = logits.gather(1, top_idx[:, :k]).tolist()
    softmax_score = softmaxes.gather(1, top_idx[:, :k]).tolist()
    class_labels = [[imagenet_classes[idx] for idx in i_row] for i_row in top_idx[:, :k]]
    for b in range(len(logits)):
        for (logit, smax, cls) in zip(logits_score[b], softmax_score[b], class_labels[b]):
            print(f'{logit:.3f} {smax:.3f} {cls}')
        print()


def resnet50_features(
    model: torch.nn.Module, video_path: str, batch_size: int,
    device: torch.device, transforms: callable, show_imagenet_pred: bool, imagenet_class_path: str,
    classifier: torch.nn.Module, features_name: str = 'resnet50'
) -> Dict[str, np.ndarray]:
    '''Loads the video for a specified video path extracts ImageNet features

    Args:
        model (torch.nn.Module): pre-trained ImageNet model
        video_path (str): path to a video
        batch_size (int): if the feature extraction should be processed in batches
        device (torch.device): GPU id
        transforms (callable): test-time transformations to apply on the input to ImageNet
        show_imagenet_pred (bool): if True, prints out the predictions for a feature on imagenet dataset
        imagenet_class_path (str): path to labels of Imagenet
        classifier (torch.nn.Module): pre-trained classification layer. Used in showing imagenet predictions
        features_name (str): name of the features. Will be used as a key in the output dict.

    Returns:
        Dict[str, np.ndarray]: 'features_nme', 'fps', 'timestamps_ms'
    '''
    def _run_on_a_batch(vid_feats, batch, model, classifier, imagenet_class_path, device):
        batch = torch.cat(batch).to(device)

        with torch.no_grad():
            batch_feats = model(batch)
            vid_feats.extend(batch_feats.tolist())
            # show predicitons on imagenet dataset (might be useful for debugging)
            if show_imagenet_pred:
                logits = classifier(batch_feats)
                show_predictions_on_IN(logits, imagenet_class_path)

    # read a video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps_ms = []
    batch = []
    vid_feats = []

    while cap.isOpened():
        frame_exists, rgb = cap.read()

        if frame_exists:
            timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            # prepare data (first -- transform, then -- unsqueeze)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = transforms(rgb)
            rgb = rgb.unsqueeze(0)
            batch.append(rgb)
            # when batch is formed to inference
            if len(batch) == batch_size:
                _run_on_a_batch(vid_feats, batch, model, classifier, imagenet_class_path, device)
                # clean up the batch list
                batch = []
        else:
            # if the last batch was smaller than the batch size
            if len(batch) != 0:
                _run_on_a_batch(vid_feats, batch, model, classifier, imagenet_class_path, device)
            cap.release()
            break

    features_with_meta = {
        f'{features_name}': np.array(vid_feats),
        'fps': np.array(fps),
        'timestamps_ms': np.array(timestamps_ms)
    }

    return features_with_meta



# if __name__ == "__main__":
#     import torchvision.models as models
#     import sys
#     import torchvision.transforms as transforms
#     sys.path.insert(0, '.')  # nopep8
#     video_path = '/home/vladimir/project4/video_features/sample/v_ZNVhz7ctTq0.mp4'
#     RESIZE_SIZE = 256
#     CENTER_CROP_SIZE = 224
#     TRAIN_MEAN = [0.485, 0.456, 0.406]
#     TRAIN_STD = [0.229, 0.224, 0.225]
#     IMAGENET_CLASS_LABELS = '/home/vladimir/project4/video_features/models/resnet50/checkpoints/IN_label_map.txt'
#     show_imagenet_pred = True
#     B = 32

#     transforms = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(RESIZE_SIZE),
#         transforms.CenterCrop(CENTER_CROP_SIZE),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD)
#     ])
#     device = torch.device('cuda:0')
#     model = models.resnet50(pretrained=True).to(device)
#     model.eval()
#     # save the pre-trained classifier for show_preds and replace it in the net with identity
#     model_class = model.fc
#     model.fc = torch.nn.Identity()

#     features = resnet50_features(
#         model, video_path, B, device, transforms, show_imagenet_pred, IMAGENET_CLASS_LABELS, model_class
#     )
#     print(features['resnet50'].shape)
