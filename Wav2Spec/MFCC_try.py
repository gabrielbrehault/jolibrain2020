# inspi : https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
# produit un spectrogramme MFCC


import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display


filename = input("Fichier : ", )
samples, sample_rate = librosa.load(filename)

S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)

# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
plt.title('Mel power spectrogram ')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()

mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

# Let's pad on the first and second deltas while we're at it
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

plt.figure(figsize=(12, 4))
librosa.display.specshow(delta2_mfcc)
# plt.ylabel('MFCC coeffs')
# plt.xlabel('Time')
# plt.title('MFCC')
# plt.colorbar()
# plt.tight_layout()

plt.axis('off')

plt.savefig(filename.split('.')[0]+'mfcc.png',
        dpi=100,
        aspect='normal',
        bbox_inches='tight',
        pad_inches=0)