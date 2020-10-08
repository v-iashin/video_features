import torch
from typing import Iterable
from PIL import Image

def resize(img, size, resize_to_smaller_edge=True, interpolation=Image.BILINEAR):
    r"""
    (v-iashin): this is almost the same implementation as in PyTorch except it has no _is_pil_image() check
    and has an extra argument governing what happens if `size` is `int`.

    Reference: https://pytorch.org/docs/1.6.0/_modules/torchvision/transforms/functional.html#resize
    Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller (bigger depending on `resize_to_smaller_edge`) edge of the image will be matched
            to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        resize_to_smaller_edge (bool, optional): if True the smaller edge is matched to number in `size`,
            if False, the bigger edge is matched to it.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if (w < h) == resize_to_smaller_edge:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

class ResizeImproved(object):

    def __init__(self, size: int, resize_to_smaller_edge: bool = True, interpolation=Image.BILINEAR):
        self.size = size
        self.resize_to_smaller_edge = resize_to_smaller_edge
        self.interpolation = interpolation

    def __call__(self, img):
        return resize(img, self.size, self.resize_to_smaller_edge, self.interpolation)

class ToTensorWithoutScaling(object):

    def __call__(self, np_img):
        return torch.from_numpy(np_img).permute(2, 0, 1).float()

class ToFloat(object):

    def __call__(self, byte_img):
        return byte_img.float()


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import numpy as np
    width = 100
    height = 200
    max_side_size = 512
    resize_to_smaller_edge = False
    if max_side_size is not None:
        transforms = transforms.Compose([
            transforms.ToPILImage(),
            ResizeImproved(max_side_size, resize_to_smaller_edge),
            transforms.PILToTensor(),
            ToFloat()
        ])
    else:
        transforms = transforms.Compose([
            ToTensorWithoutScaling()
        ])
    a = np.random.randint(0, 255, (height, width, 3)).astype(np.uint8)
    print(a.shape)
    b = transforms(a)
    print(b)
    print(b.shape)
    # print(b.size)
    # print(b.width)
    # assert b.height == max_side_size
    # b.show()
