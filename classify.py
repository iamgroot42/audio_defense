## classify.py -- actually classify a sequence with DeepSpeech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## Modified by Anshuman Suri <anshuman14021@iiitd.ac.in>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
import argparse

import scipy.io.wavfile as wav

import struct
import time
import os
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")

try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True


class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]


class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v

tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None


from util.text import ctc_label_dense_to_sparse
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"
modelInitDone = False


def getAudioPrediction(sess, audio):
    global modelInitDone
    N = len(audio)
    new_input = tf.placeholder(tf.float32, [1, N])
    lengths = tf.placeholder(tf.int32, [1])
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        logits = get_logits(new_input, lengths)

    if not modelInitDone:
        init(sess)
        modelInitDone = True

    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)
    length = (len(audio)-1)//320
    l = len(audio)
    r = sess.run(decoded, {new_input: [audio], lengths: [length]})
    tts = "".join([toks[x] for x in r[0].values])    
    return tts


def init(sess):
    saver = tf.train.Saver()
    saver.restore(sess, "models/session_dump")


def main():
    # Create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    # Run classification for given audio samples
    for filename in sys.argv[1:]:
        if filename.split(".")[-1] == 'wav':
            _, audio = wav.read(filename)
        else:
            raise Exception("Unknown file format")
        print(getAudioPrediction(sess, audio))


if __name__ == "__main__":  
    main()
