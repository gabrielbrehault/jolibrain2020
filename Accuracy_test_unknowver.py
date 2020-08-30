import subprocess
from pathlib import Path
from os import listdir
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os.path

from Wav2Exr_returnver import Wav2Exr

# pour ajouter la voice activity detection :

# from Vad2File import vad2file


classes = ["zero", "one", "two", "three", "four", "five", "six",
            "seven", "eight", "nine", "yes", "no", "up", "down",
            "left", "right", "on", "off", "stop", "go"]

host = 'http://localhost:8080/predict'

perf = []
total = 0
success = 0
class_total = 0
n_unknow = 0

if not os.path.isdir('output'):
    Path('output').mkdir()

for classe in tqdm(classes):
    perf.append(0)
    path = Path(f'/home/gabri/JoliBrain/SpeechRecog/test_data/{classe}')
    for wavfile in (f for f in listdir(path)):

        if wavfile.split('.')[-1] != 'wav':
            continue


        # pour ajouter la voice activity detection :

        # vad2file(f'{path / Path(wavfile)}', 3)

        # if not os.path.isfile('chunk.wav'):
        #     continue
        


        # ATTENTION : si la partie au-dessus n'est pas commentée, 
        # il faut commenter la ligne ci-dessous et décommenter la suivante

        output_list = Wav2Exr(Path(f'{path / Path(wavfile)}'))
        # output_list = Wav2Exr(Path('chunk.wav'))

        for exrfile in output_list:

            class_total += 1
            total += 1

            # print("\n Classe évaluée :", classe)

            param = {
                "service": "speech",
                "parameters": {
                    "output": {
                        "best": 3
                        },
                    "mllib": {
                        "gpu": True
                        }
                    },
                "data": [
                    f"{Path(exrfile)}"
                    ]
                }

            response = requests.post(host, json=param)
            cat = response.json()['body']['predictions'][0]['classes'][0]['cat']
            prob = response.json()['body']['predictions'][0]['classes'][0]['prob']

            if float(prob) < 0.6:
                n_unknow += 1
                continue

            if cat == classe :
                success += 1
                perf[-1] += 1
            
        subprocess.call("rm /home/gabri/JoliBrain/SpeechRecog/output/*", shell=True)

        # à décommenter pour la vad :

        # subprocess.call("rm chunk.wav", shell=True)

    perf[-1] /= class_total
    class_total = 0

accuracy = success/total

y_pos = np.arange(len(classes))

plt.figure(figsize=(20, 10))
plt.bar(y_pos, perf)
plt.xticks(y_pos, classes)
plt.title(f'Accuracy = {accuracy} \n Number of unknow class = {n_unknow} \n Class accuracy :')

plt.savefig('class_accuracy.png')
plt.show()

print("Accuracy = ", accuracy)