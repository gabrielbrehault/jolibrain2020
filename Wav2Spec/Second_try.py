# inspi : https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
# utilise scipy.signal pour réaliser le spectrogramme


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa


def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


filename = input("Fichier : ", )

#sample_rate, samples = wavfile.read(filename)

# La ligne au-dessus donne un résultat similaire aux autres 
# try mais ne fonctionne qu'en mono.
# La ligne au-dessous fonctionne aussi avec du stéréo mais 
# donne un résultat différend

samples, sample_rate = librosa.load(filename)

freqs, times, spectrogram = log_specgram(samples, sample_rate)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot()
ax1.imshow(spectrogram.T, aspect='auto', origin='lower', extent=[times.min(), times.max(), freqs.min(), freqs.max()])
plt.axis('off')
# ax1.set_yticks(freqs[::16])
# ax1.set_xticks(times[::16])
# ax1.set_title('Spectrogram of ' + filename)
# ax1.set_ylabel('Freqs in Hz')
# ax1.set_xlabel('Seconds')

plt.savefig(filename.split('.')[0]+'2.png',
        dpi=100, # Dots per inch,
        aspect='normal',
        bbox_inches='tight',
        pad_inches=0)