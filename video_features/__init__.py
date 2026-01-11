"""Video features extraction package.

This package provides utilities for extracting features from video clips.
The main packages are:
- models: Feature extraction models
- utils: Utility functions
"""

__version__ = "0.1.0"

# Expose subpackages
from video_features import models
from video_features import utils

__all__ = ['models', 'utils', '__version__']

