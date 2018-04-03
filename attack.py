## attack.py -- generate audio adversarial examples
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## Edited by Anshuman Suri <anshuman14021@iiitd.ac.in>
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
from shutil import copyfile

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


class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=5000, batch_size=1,
                 l2penalty=float('inf')):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """
        
        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't clobber them.
        self.prefix = "qq_"
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name=self.prefix + 'delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name=self.prefix + 'mask')
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name=self.prefix + 'cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name=self.prefix + 'original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name=self.prefix + 'lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name=self.prefix + 'importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name=self.prefix + 'phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name=self.prefix + 'phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name=self.prefix + 'phrase_lengths')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -2000, 2000)*self.rescale

        # We set the new input to the model to be the abve delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = self.apply_delta*mask + original

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape,
                                 stddev=2)
        pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in, lengths)

        # And finally restore the graph to make the classifier actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if self.prefix not in x.name])
        saver.restore(sess, "models/session_dump")

        # Choose the loss function we want -- either CTC or CW
        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths, batch_size)
            
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)

            # Slight hack: an infinite l2 penalty means that we don't penalize l2 distortion
            # The code runs faster at a slight cost of distortion, and also leaves one less
            # paramaeter that requires tuning.
            if not np.isinf(l2penalty):
                loss = tf.reduce_mean((self.new_input-self.original)**2,axis=1) + l2penalty*ctcloss
            else:
                loss = ctcloss
            self.expanded_loss = tf.constant(0)
            
        elif loss_fn == "CW":
            raise NotImplemented("The current version of this project does not include the CW loss function implementation.")
        else:
            raise

        self.loss = loss
        self.ctcloss = ctcloss
        
        # Set up the Adam optimizer to perform gradient descent for us
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)

        grad,var = optimizer.compute_gradients(self.loss, [delta])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])
        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        sess.run(tf.variables_initializer(new_vars+[delta]))

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

    def attack(self, audio, lengths, target, finetune=None):
        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths)-1)//320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths)-1)//320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None]*self.batch_size
        
        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iterations
        for i in range(MAX):
            iteration = i
            now = time.time()

            
            # Print out some debug information every 10 iterations.
            if i%10 == 0:
                new, delta, r_out, r_logits = sess.run((self.new_input, self.delta, self.decoded, self.logits))
                lst = [(r_out, r_logits)]
            
                for out, logits in lst:
                    chars = out[0].values
            
                    res = np.zeros(out[0].dense_shape)+len(toks)-1
                
                    for ii in range(len(out[0].values)):
                        x,y = out[0].indices[ii]
                        res[x,y] = out[0].values[ii]
            
                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    print("\n".join(res))
                    
                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))
            
            feed_dict = {}
                
            # Actually do the optimization step
            d, el, cl, l, logits, new_input, _ = sess.run((self.delta, self.expanded_loss,
                                                           self.ctcloss, self.loss,
                                                           self.logits, self.new_input,
                                                           self.train),
                                                          feed_dict)
                    
            # Report progress
            print("%.3f"%np.mean(cl), "\t", "\t".join("%.3f"%x for x in cl))

            logits = np.argmax(logits,axis=2).T
            for ii in range(self.batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                if (self.loss_fn == "CTC" and i%10 == 0 and res[ii] == "".join([toks[x] for x in target[ii]])) \
                   or (i == MAX-1 and final_deltas[ii] is None):
                    # Get the current constant
                    rescale = sess.run(self.rescale)
                    if rescale[ii]*2000 > np.max(np.abs(d)):
                        # If we're already below the threshold, then
                        # just reduce the threshold to the current
                        # point and save some time.
                        print("It's way over", np.max(np.abs(d[ii]))/2000.0)
                        rescale[ii] = np.max(np.abs(d[ii]))/2000.0

                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.
                    rescale[ii] *= .8

                    # Adjust the best solution found so far
                    final_deltas[ii] = new_input[ii]

                    print("Worked i=%d ctcloss=%f bound=%f"%(ii,cl[ii], 2000*rescale[ii][0]))
                    sess.run(self.rescale.assign(rescale))

        return final_deltas


def init(sess, phrase, maxlen, batch_size=2, lr=100, iterations=1000, l2penalty=float('inf')):
    attack = Attack(sess, 'CTC', len(phrase), maxlen,
                        batch_size=batch_size,
                        learning_rate=lr,
                        num_iterations=iterations,
                        l2penalty=l2penalty)
    return attack


def runAttacks(sess, attack, inputAudios, phrase, maxlen):
    lengths = []
    audios = []
    for audio in inputAudios:
        lengths.append(len(audio))
        audios.append(list(audio))
    audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])    
    deltas = attack.attack(audios,
                           lengths,
                           [[toks.index(x) for x in phrase]]*len(audios))
    return deltas


def main():
    # Dynamic GPU growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True    
    sess = tf.Session(config=config)

    # Load the inputs that we're given
    audios = []
    lengths = []
    for filename in sys.argv[1:]:
        fs, audio = wav.read(filename)
        audios.append(audio)
        lengths.append(len(audio))
        assert fs == 16000
        print('source dB', 20*np.log10(np.max(np.abs(audio))))

    # Create object of attack class
    phrase = "i am a bad boy"
    maxlen = max(map(len, audios))
    print("Maximum length", maxlen)
    attack = init(sess, phrase, maxlen, batch_size=2, lr=100, iterations=1000, l2penalty=float('inf'))

    # Run attack on given audio files
    deltas = runAttacks(sess, attack, audios, phrase)

    # And now save it to the desired output
    for i in range(len(sys.argv[1:])):
        wav.write(path, 16000,
                  np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                   -2**15, 2**15-1),dtype=np.int16))
        print("Final distortion", np.max(np.abs(deltas[i][:lengths[i]]-audios[i][:lengths[i]])))


if __name__ == "__main__":
    main()
