import torch
import hashlib
from pathlib import Path


def md5sum(path: str):
    '''The same as `md5sum <file>` in Linux'''
    with open(path, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def make_ref(args, video_path: Path, data, save_path):
    assert not save_path.exists()
    save_path.parent.mkdir(exist_ok=True)
    to_save = {
        'args': args,
        'video_path': video_path,
        'video_path_md5': md5sum(video_path),
        'data': data,
    }
    torch.save(to_save, save_path)
    print('Saved to', save_path)
