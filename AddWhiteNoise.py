import numpy as np
import random
import librosa
import sys
import wave

from scipy.io import wavfile


def load_audio_file(file_path):
    input_length = 16000
    data = librosa.core.load(file_path)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def AddWhiteNoise(audiofile):

    sr, data = wavfile.read(audiofile)
    # data, sr = load_audio_file(audiofile), 16000

    wn = np.random.randn(len(data))
    data_wn = data + 0.005*wn

    return data_wn, sr

if __name__=='__main__':
    wn_ver, sr = AddWhiteNoise(sys.argv[1])
    new_file = 'test_wnver.wav'

    with wave.open(sys.argv[1], 'rb') as wf:
        sampwidth = wf.getsampwidth()

    with wave.open(new_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(wn_ver)    