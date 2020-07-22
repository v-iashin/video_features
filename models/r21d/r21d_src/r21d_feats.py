import torch
import torch.nn.functional as F
from torchvision.io import read_video
from models.r21d.utils.utils import form_slices


def show_predictions_on_K400(logits: torch.FloatTensor, kinetics_class_path: str):
    '''Prints out predictions using logits extracted from one feature window

    Args:
        logits (torch.FloatTensor): after-classification layer vector
        kinetics_class_path (str): path to labels of Kinetics 400
    '''
    softmaxes = F.softmax(logits, dim=-1)

    # Show predictions
    top_val, top_idx = torch.sort(softmaxes, dim=-1, descending=True)
    kinetics_classes = [x.strip() for x in open(kinetics_class_path)]
    for i in range(5):
        logits_score = logits[top_idx[i]].item()
        print(f'{logits_score:.3f} {top_val[i]:.3f} {kinetics_classes[top_idx[i]]}')
    print()


def r21d_features(
    model: torch.nn.Module, video_path: str, feature_size: int, stack_size: int, step_size: int, 
    device: torch.device, transforms: callable, show_kinetics_pred: bool, kinetics_class_path: str, 
    classifier: torch.nn.Module
) -> torch.FloatTensor:
    '''Loads the video for a specified video path, generates a slicing pattern given the stack and step sizes,
    extracts R(2+1)D features

    Args:
        model (torch.nn.Module): pre-trained R(2+1)D model
        video_path (str): path to a video
        feature_size (int): the expected feature size
        stack_size (int): number of frames to use for to extract one feature
        step_size (int): number of frames to step for a windows
        device (torch.device): GPU id
        transforms (callable): test-time transformations to apply on the input to R(2+1)D
        show_kinetics_pred (bool): if True, prints out the predictions for a feature on Kinetics 400 dataset
        kinetics_class_path (str): path to labels of Kinetics 400
        classifier (torch.nn.Module): pre-trained classification layer. Used in showing Kinetics predictions

    Returns:
        torch.FloatTensor: features for the video
    '''
    # read a video
    rgb, audio, info = read_video(video_path, pts_unit='sec')
    # prepare data (first -- transform, then -- unsqueeze)
    rgb = transforms(rgb)
    rgb = rgb.unsqueeze(0)
    # slice the
    slices = form_slices(rgb.size(2), stack_size, step_size)
    r21d_feats = torch.zeros((len(slices), feature_size), device=device)

    for stack_idx, (start_idx, end_idx) in enumerate(slices):
        # inference
        with torch.no_grad():
            r21d_feats[stack_idx] = model(rgb[:, :, start_idx:end_idx, :, :].to(device))
            # show predicitons on kinetics dataset (might be useful for debugging)
            if show_kinetics_pred:
                logits = classifier(r21d_feats[stack_idx])
                print(f'{video_path} @ frames ({start_idx}, {end_idx})')
                show_predictions_on_K400(logits, kinetics_class_path)
    
    return r21d_feats
