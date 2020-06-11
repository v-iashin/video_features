import os
import subprocess

from utils.utils import which_ffmpeg

def extract_wav_from_mp4(video_path: str, tmp_path: str) -> str:
    '''Extracts .wav file from .aac which is extracted from .mp4
    We cannot convert .mp4 to .wav directly. For this we do it in two stages: .mp4 -> .aac -> .wav

    Args:
        video_path (str): Path to a video
        audio_path_wo_ext (str): 

    Returns:
        [str, str] -- path to the .wav and .aac audio
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'

    # extract video filename from the video_path
    video_filename = os.path.split(video_path)[-1].replace('.mp4', '')

    # the temp files will be saved in `tmp_path` with the same name 
    audio_aac_path = os.path.join(tmp_path, f'{video_filename}.aac')
    audio_wav_path = os.path.join(tmp_path, f'{video_filename}.wav')

    # constructing shell commands and calling them
    mp4_to_acc = f'{which_ffmpeg()} -hide_banner -loglevel panic -y -i {video_path} -acodec copy {audio_aac_path}'
    aac_to_wav = f'{which_ffmpeg()} -hide_banner -loglevel panic -y -i {audio_aac_path} {audio_wav_path}'
    subprocess.call(mp4_to_acc.split())
    subprocess.call(aac_to_wav.split())

    return audio_wav_path, audio_aac_path
