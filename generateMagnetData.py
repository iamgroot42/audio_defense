import sys
import os
import time
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf

import attack as att


if __name__ == "__main__":

	# Output directory
	out_dir = 'perturbed_subset'
	source_dir = 'commonvoice_subset'
	phrase_index = int(sys.argv[1])
	batch_size = 5

	# Load target phrases
	target_phrases = []
	with open('target_phrases.txt', 'r') as f:
		for phrase in f:
			target_phrases.append(phrase.rstrip('\n'))
	assert(len(target_phrases) > phrase_index)
	target_phrase = target_phrases[phrase_index]

	# Create session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)

	# Load audio samples
	audios = []
	lengths = []
	for file in os.listdir(source_dir):
		fs, audio = wav.read(os.path.join(source_dir, file))
		audios.append(audio)
		lengths.append(len(audio))
		assert fs == 16000

	maxlen = max(map(len, audios))
	
	# Create attack object
	attack = att.init(sess, target_phrase, maxlen, batch_size=batch_size, lr=100, iterations=1000, l2penalty=float('inf'))	

	# Run attack
	deltas = []
	for i in range(0, len(audios), batch_size):
		audio_batch = audios[i: i + batch_size]
		deltas += att.runAttacks(sess, attack, audio_batch, target_phrase, maxlen)

	# Save perturbed audio files
	base = time.time()
	for i, delta in enumerate(deltas):
		wav.write(os.path.join(out_dir, str(base) + "_" + str(i) + ".wav"), 16000,
			np.array(np.clip(np.round(deltas[i][:lengths[i]]), -2**15, 2**15-1),dtype=np.int16))
