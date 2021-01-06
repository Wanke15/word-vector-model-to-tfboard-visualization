#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser(description='Convert word2vec model to tensorboard model for visualization')

parser.add_argument('--wvmodel', default='model.bin', help="Trained word vector model", type=str)
parser.add_argument('--outdir', default='tensorboard', help="Output tensorboard path", type=str)

args = parser.parse_args()

import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
import gensim
import os

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

fname = args.wvmodel
if fname.endswith(".bin"):
    model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(fname, binary=True)
elif fname.endswith(".txt"):
    model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(fname, binary=False)
else:
    raise ValueError("Expect word2vec model format are: 'bin' or 'txt' but got: {}".format(fname))


max_size = len(model.wv.vocab)-1


w2v = np.zeros((max_size,model.vector_size))


path = args.outdir

if not os.path.exists(path):
    os.mkdir(path)

with open(os.path.join(path, "metadata.tsv"), 'w+') as file_metadata:
    for i,word in enumerate(model.index2word[:max_size]):
        w2v[i] = model[word]
        file_metadata.write(word + '\n')


sess = tf.InteractiveSession()


#Let us create a 2D tensor called embedding that holds our embeddings.
with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v, trainable=False, name='embedding')


tf.global_variables_initializer().run()


# let us create an object to Saver class which is actually used to
#save and restore variables to and from our checkpoints
saver = tf.train.Saver()


# using file writer, we can save our summaries and events to our event file.
writer = tf.summary.FileWriter(path, sess.graph)


# adding into projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'embedding'
embed.metadata_path = 'metadata.tsv'


# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)

saver.save(sess, path+'/model.ckpt', global_step=max_size)
print("Done!")
