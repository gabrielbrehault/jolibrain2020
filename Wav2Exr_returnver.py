import logging
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

import cv2
import librosa

import sys


def make_slice(total: int, size: int, step: int) -> Iterator[slice]:
    """
    Sliding window over the melody. step should be less than or equal to size.
    """
    if step > size:
        logging.warn("step > size, you probably miss some part of the melody")
    if total < size:
        yield slice(0, total)
        return
    for t in range(0, total - size, step):
        yield slice(t, t + size)
    if t + size < total:
        yield slice(total - size, total)


def Wav2Exr(filepath):

    y, sr = librosa.load(Path(filepath))

    D = librosa.stft(y, 2 ** 9, center=True)
    spec = librosa.amplitude_to_db(
        librosa.magphase(D)[0], ref=np.max
        )

    output_list = []

    for slc in make_slice(spec.shape[1], 257, 100):
        pattern = f"{slc.start:>05d}_{slc.stop:>05d}.exr"
        cv2.imwrite((Path('output') / pattern).as_posix(), spec[:, slc])
        output_list.append((Path('/home/gabri/JoliBrain/SpeechRecog/output') / pattern).as_posix())

    return output_list


if __name__=='__main__':
    Wav2Exr(sys.argv[1])
