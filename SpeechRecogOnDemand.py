from Wav2Exr_returnver import Wav2Exr
from Vad2File_2 import vad2file

from pathlib import Path
import time
import subprocess
import sys
import requests
import time
import os.path


host = 'http://localhost:8080/predict'
path = Path(sys.argv[1])

while True:
    try:
        if input("Appuyer sur entrée pour donner une commande, taper 'q' pour quitter :", ) == 'q':
            break
        
        proc = subprocess.Popen(['arecord', path])
        time.sleep(3)

        vad2file(f'{path}', 1)

        if not os.path.isfile('chunk.wav'):
            print("Sorry, I didn't understand.")
            continue

        output_list = Wav2Exr(Path('chunk.wav'))
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
            prob = response.json()['body']['predictions'][0]['classes'][0]['prob']
            if prob > 0.4 :
                print(f"You said : {cat}")
            else:
                print("Sorry, I didn't understand.")

    finally:
        proc.kill()
        if os.path.isfile('chunk.wav'):
            subprocess.call("rm /home/gabri/JoliBrain/SpeechRecog/output/*", shell=True)
            subprocess.call("rm /home/gabri/JoliBrain/SpeechRecog/chunk.wav", shell=True)
            subprocess.call(f"rm /home/gabri/JoliBrain/SpeechRecog/{path}", shell=True)