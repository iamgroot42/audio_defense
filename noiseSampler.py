import scipy.io.wavfile as wav
import sys
import numpy as np


fs, audio = wav.read(sys.argv[1])

for i in range(1, 11):
	noise = np.random.normal(0, 1, len(audio))
	perturbed_audio = audio + noise * i
	perturbed_audio = np.array(np.clip(np.round(perturbed_audio), -2**15, 2**15-1), dtype=np.int16)
	wav.write("newAdv" + str(i) + ".wav", fs, perturbed_audio)
