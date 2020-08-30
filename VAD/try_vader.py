import vader
import sys

# use your own mono, preferably 16kHz .wav file
filename = sys.argv[1]

# returns segments of vocal activity (unit: seconds)
# note: it uses a pre-trained logistic regression by default
segments = vader.vad(filename, threshold=0.7, window=20, method="nn")

# where to dump audio files
out_folder = "segments"
# write segments into .wav files
vader.vad_to_files(segments, filename, out_folder)