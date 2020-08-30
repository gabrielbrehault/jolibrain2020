import subprocess
from pathlib import Path
from os import listdir
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np



classes = ["zero", "one", "two", "three", "four", "five", "six",
            "seven", "eight", "nine", "yes", "no", "up", "down",
            "left", "right", "on", "off", "stop", "go"]

host = 'http://localhost:8080/predict'

perf = []
total = 0
success = 0
class_total = 0

for classe in tqdm(classes):
    perf.append(0)
    path = Path(f'/home/gabri/JoliBrain/SpeechRecog/test_data/{classe}/output')
    for wavfile in (f for f in listdir(path)):

        class_total += 1

        total += 1

        #print("\n Classe évaluée :", classe)

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
                f"{path / Path(wavfile)}"
                ]
            }

        response = requests.post(host, json=param)
        cat = response.json()['body']['predictions'][0]['classes'][0]['cat']
        if cat == classe :
            success += 1
            perf[-1] += 1

    perf[-1] /= class_total
    class_total = 0

y_pos = np.arange(len(classes))

plt.figure(figsize=(20, 10))
plt.bar(y_pos, perf)
plt.xticks(y_pos, classes)
plt.title('Class accuracy')

#plt.savefig('class_accuracy.png')
plt.show()

accuracy = success/total
print("Accuracy = ", accuracy)