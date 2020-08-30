from os import listdir
import requests
from pathlib import Path
import subprocess

from Wav2Exr_returnver import Wav2Exr


host = 'http://localhost:8080/predict'
path = Path('/home/gabri/JoliBrain/Speech_recognition/input/test/audio')

fail = 0
success = 0
total = 0

for wavfile in (f for f in listdir(path)):

    subprocess.run(['aplay', f'{path / Path(wavfile)}'])

    output_list = Wav2Exr(Path(f'{path / Path(wavfile)}'))

    for exrfile in output_list:

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
        print(cat)

    subprocess.run("rm /home/gabri/JoliBrain/SpeechRecog/output/*", shell=True)
    
    total += 1

    if input('Bon ? Appuyer sur entrée si oui, 0 sinon :',) == str(0):
        pass
    else :
        success +=1

    print('Accuracy =', success/total)
