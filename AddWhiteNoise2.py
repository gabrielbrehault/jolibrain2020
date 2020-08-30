import numpy as np
from scikits.audiolab import wavread, wavwrite
import sys

data1, fs1, enc1 = wavread(sys.argv[1])
data2, fs2, enc2 = wavread(sys.argv[2])

assert fs1 == fs2
assert enc1 == enc2
result = 0.5 * data1 + 0.5 * data2

wavwrite(result, 'result.wav')