import classify
import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np


def perturbedOutputs(audio, noiseLevels):
	noisy = []
	for n in noiseLevels:
		noise = np.random.normal(0, 1, len(audio))
		noisy_audio = audio + noise * n
		noisy_audio = np.array(np.clip(np.round(noisy_audio), -2**15, 2**15-1), dtype=np.int16)
		noisy.append(noisy_audio)
	return noisy


def wer(r, h):
        d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
        d = d.reshape((len(r)+1, len(h)+1))
        for i in range(len(r)+1):
                for j in range(len(h)+1):
                        if i == 0:
                                d[0][j] = j
                        elif j == 0:
                                d[i][0] = i

        # computation
        for i in range(1, len(r)+1):
                for j in range(1, len(h)+1):
                        if r[i-1] == h[j-1]:
                                d[i][j] = d[i-1][j-1]
                        else:
                                substitution = d[i-1][j-1] + 1
                                insertion    = d[i][j-1] + 1
                                deletion     = d[i-1][j] + 1
                                d[i][j] = min(substitution, insertion, deletion)

        return d[len(r)][len(h)]


def constructFeatures(sentences):
	splits = [x.split(' ') for x in sentences]
	features = []
	for i in range(1, len(splits)):
		features.append(wer(splits[i], splits[i-1]))
	return features


if __name__ == "__main__":
	import sys
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	noiseLevels = range(0, 11, 2)
	_, audio = wav.read(sys.argv[1])
	noisy = perturbedOutputs(audio, noiseLevels)
	outputs = [ classify.getAudioPrediction(sess, x) for x in noisy]
	print(constructFeatures(outputs))
