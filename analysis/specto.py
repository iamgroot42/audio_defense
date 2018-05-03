import scipy.io.wavfile as wav
from scipy import signal

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys

fs, x = wav.read(sys.argv[1])
f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
import numpy as np
plt.ylim(0, 2000)
plt.savefig(sys.argv[2])
