import sys
import matplotlib
matplotlib.use('Agg')

from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read(sys.argv[1])
times = np.arange(len(data))/float(samplerate)

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
plt.fill_between(times, data, color='k')#data[:,0], data[:,1], color='k') 
plt.xlim(0.5, times[-1]) #times[0], times[-1])
#plt.xlim(0.75, 1.25)
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.savefig(sys.argv[2], dpi=100)
#plt.show()
