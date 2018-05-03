import defense
# import defenseneg
import classify
import scipy.io.wavfile as wav
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import os

# Paths to perturbed/unperturbed audio samples
cleanDir = './commonvoice_subset2'
perturbedDir = './perturbed_subset'

# Create session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# Define noise ranges that are to be considered
noiseLevels = range(0, 11, 2)

# Create data to train distinguishing model
X = []
Y = []

_, audio = wav.read(sys.argv[1])
noisy = defense.perturbedOutputs(audio, noiseLevels)
outputs = [ classify.getAudioPrediction(sess, x) for x in noisy]
newaud = classify.getAudioPrediction(sess, audio)
print sys.argv[1]
print newaud
for i in outputs:
	print i
# features = defense.constructFeatures(outputs)

# print(" ".join([str(x) for x in features]))
