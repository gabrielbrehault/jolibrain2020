import wave 

import sys

from sample_viz import sample_viz

def sample_cut(filename):

    with wave.open(sys.argv[1], 'rb') as wf:
        sampwidth = wf.getsampwidth()

    samples, sample_rate = sample_viz(filename)

    cut_ver = samples[int(input("Keep samples from :")): int(input("To :"))]

    new_file = filename.split('.')[0] + '_cutver.wav'

    with wave.open(new_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(cut_ver)

if __name__=='__main__':
    sample_cut(sys.argv[1])