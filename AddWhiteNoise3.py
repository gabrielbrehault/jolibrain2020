from pydub import AudioSegment
import sys
from pathlib import Path
from sample_viz import sample_viz

def AddNoise(audio, noise):

    sound1 = AudioSegment.from_file(audio)
    sound2 = AudioSegment.from_file(noise)

    combined = sound1.overlay(sound2)

    combined.export("./wnver3.wav", format='wav')

if __name__=='__main__':
    AddNoise(Path(sys.argv[1]), Path(sys.argv[2]))
    sample_viz("wnver3.wav")