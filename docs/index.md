<figure>
  <img src="_assets/base.png" width="300" />
</figure>

This is the documentation for `video_features`. A library (😅) that allows you to extract features from raw videos using the pre-trained nets. So far, it supports several extractors that capture visual appearance, calculates optical flow, and, even, audio features. The source code lives at [v-iashin/video_features](https://github.com/v-iashin/video_features).

The source code was intended to support the feature extraction pipeline for two of my papers ([BMT](https://arxiv.org/abs/2005.08271) and [MDVC](https://arxiv.org/abs/2003.07758)). This library (😅) somehow emerged out of that code and now has more models implemented.

---

## Supported models
*If you would like to see more, please create an [Issue](https://github.com/v-iashin/video_features/issues).*

Action Recognition

- [I3D-Net RGB + Flow (Kinetics 400)](models/i3d.md)
- [R(2+1)d RGB (Kinetics 400)](models/r21d.md)

Sound Recognition

- [VGGish (AudioSet)](models/vggish.md)

Optical Flow

- [RAFT (FlyingChairs, FlyingThings3D, Sintel)](models/raft.md)
- [PWC-Net (Sintel)](models/pwc.md)

Image Recognition

- [ResNet-18,34,50,101,152 (ImageNet)](models/resnet.md)
