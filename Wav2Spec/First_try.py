# inspi : https://github.com/iamvishnuks/Audio2Spectrogram/blob/master/Audio2Spectrogram.py
# utilise matplotlit.pyplot.specgram pour réaliser le spectrogramme
# version très simplifiée

from matplotlib import pyplot as plt
from PIL import Image
from scipy.io import wavfile
import numpy as np
import sys


def Stereo2Mono(data):
    return (data.sum(axis=1)/2)


def Wav2Spec(audio_file):
    '''
    Crée un spectrogramme en .png du fichier audio dans le même dossier
    à partir d'un .wav en mono ou stéréo.

    '''

    _, data = wavfile.read(str(audio_file))

    nfft = 256    # Length of the windowing segments
    fs = 256      # Sampling frequency


    # problème ici : la fonction plt.specgram() ne fonctionne correctement 
    # seulement si data est de dimesion simple (soit seulement si l'audio
    # est en mono).


    if len(data.shape) == 2:
        data = Stereo2Mono(data)

    pxx, freqs, bins, im = plt.specgram(data, nfft,fs)

    print("pxx : ", len(pxx))
    print("freqs : ", len(freqs))
    print("bins : ", len(bins))

    plt.axis('off')
    # Spectrogram saved as a .png
    plt.savefig(audio_file.split('.')[0]+'1.png',
            dpi=100, # Dots per inch,
            aspect='normal',
            bbox_inches='tight',
            pad_inches=0) 
    
    im=Image.open(audio_file.split('.')[0]+'1.png')
    rgb_im = im.convert('RGB')
    rgb_im.save(audio_file.split('.')[0]+'1.png')
    
    #return rgb_im
    
if __name__=='__main__':
     Wav2Spec(sys.argv[1])