# fonction capable de produire des spectrogrammes à partir d'un stream audio en live
# problème de lenteur

import pyaudio
import time
import wave
import sys

from matplotlib import pyplot as plt
from PIL import Image
from scipy.io import wavfile
import numpy as np

import subprocess

CHUNK = 8192

sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
FS = 44100  # Record at 44100 samples per second


def Stereo2Mono(data):
    return (data.sum(axis=1)/2)


def Wav2Spec(audio_file, index):
    '''
    Crée un spectrogramme en .png du fichier audio dans le même dossier
    à partir d'un .wav en mono ou stéréo.

    '''

    output_filename = './spec_list/'+f'{index}.png'

    _, data = wavfile.read(str(audio_file))

    nfft = 256    # Length of the windowing segments
    fs = 256      # Sampling frequency


    if len(data.shape) == 2:
        data = Stereo2Mono(data)

    _, _, _, im = plt.specgram(data, nfft,fs)

    plt.axis('off')
    # Spectrogram saved as a .png
    plt.savefig(output_filename,
            dpi=100, # Dots per inch,
            aspect='normal',
            bbox_inches='tight',
            pad_inches=0) 
    
    im=Image.open(output_filename)
    rgb_im = im.convert('RGB')
    rgb_im.save(output_filename)
    
    #return rgb_im


def StreamAudio2Spec(filepath):

    wf = wave.open(filepath, 'rb')

    # instantiate PyAudio
    p = pyaudio.PyAudio()

    # open stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

    # read data
    data = wf.readframes(CHUNK)

    #play stream
    index = 0
    while len(data) > 0:
        index += 1
        filename = './frames/'+f'{index}.wav'
        #filename = './frame.wav'
        WF = wave.open(filename, 'wb')
        WF.setnchannels(channels)
        WF.setsampwidth(p.get_sample_size(sample_format))
        WF.setframerate(FS)
        WF.writeframes(data)
        WF.close()
        Wav2Spec(filename, index)
        #Wav2Spec(filename, 0)
        stream.write(data)
        data = wf.readframes(CHUNK)

    # stop stream
    stream.stop_stream()
    stream.close()

    # close PyAudio
    p.terminate()

if __name__=='__main__':
    try:
        proc = subprocess.Popen(['arecord', sys.argv[1]])
        time.sleep(1)
        StreamAudio2Spec(sys.argv[1])
    finally:
        proc.kill()