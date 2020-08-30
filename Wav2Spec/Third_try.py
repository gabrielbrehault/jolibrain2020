# inspi : https://github.com/kjanjua26/Sound_Classification_Spectrograms/blob/master/GunShot/generateSpec.py
# utilise pylab pour réaliser le spectrogramme
# problème : le spectrogramme pour les wav en stéréo sont symétriques selon un axe horizontal

import wave
import pylab


def Wav2Spec(audio_file):

    filename = audio_file.split('.')[0]

    sound_info, frame_rate = get_wav_info(audio_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(f'{filename}3.png')
    pylab.close()

def get_wav_info(audio_file):

    wav = wave.open(audio_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

if __name__ == '__main__':
    Wav2Spec(input('Fichier audio : ',))