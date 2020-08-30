from pathlib import Path
import matplotlib.pyplot as plt

import sys

from scipy.io import wavfile

def sample_viz(filename):

    sample_rate, samples = wavfile.read(filename)

    plt.figure(figsize = (10,7))
    plt.plot(samples)

    plt.xlabel('sample')
    plt.grid()
    plt.show()
    return(samples, sample_rate)

if __name__=='__main__':
    sample_viz(Path(sys.argv[1]))