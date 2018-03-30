import os
import numpy as np
import tensorflow as tf

import attack


if __name__ == "__main__":
	# Load target phrases
	target_phrases = []
	with open('target_phrases.txt', 'r') as f:
		for phrase in f:
			target_phrases.append(phrase.rstrip('\n'))
	print(target_phrases)

	# Create session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)

	# Load audio samples
	maxlen = max(map(len, audios))
		
