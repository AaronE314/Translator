
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import sys
import ast

# from functions import create_dataset, evaluate, load_dataset, loss_function, max_length
# from functions import plot_attention, preprocess_sentence, tokenize, train_step
# from functions import train_test_split, translate, unicode_to_asciii

from Decoder import Decoder
from BahdanauAttention import BahdanauAttention
from Encoder import Encoder

# from functions import BATCH_SIZE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#####################################################################################################
# Tutorial from                                                                                     #
# https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention                                 #
#####################################################################################################

#####################################################################################################
# Functions
#####################################################################################################


def unicode_to_asciii(s):
    '''
    Converts the unicode file to ascii
    '''
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):

    w = unicode_to_asciii(w.lower().strip())

    # Creating a space between a word and the punctuation following it
    # eg: "he is a bow." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # Replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end tocken to the sentence
    # so that the model knows when to start and stop predicting

    w = '<start> ' + w + ' <end>'

    return w


def create_dataset(path, num_examples):
    '''
    1. Remove the accents
    2. Clean the sentences
    3. Return word pairs in the fromat: [ENGLISH, SPANISH]
    '''
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split(
        '\t')] for l in lines[:num_examples]]

    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters=''
    )
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):

    # Creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def loss_function(real, pred):

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def evaluate(sentence):

    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):

        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # Storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # The predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()


def translate(sentence, print=False, plot=False):

    result, sentence, attention_plot = evaluate(sentence)

    if print:
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))

    if plot:
        attention_plot = attention_plot[:len(
            result.split(' ')), :len(sentence.split(' '))]
        plot_attention(attention_plot, sentence.split(' '), result.split(' '))

    return result.replace('<end>', '')


def train():

    for epoch in range(EPOCHS):

        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))

        # Saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            # checkpoint.save(file_prefix=checkpoint_prefix)
            checkpoint_manager.save()

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:

        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims(
            [targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):

            # Passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(
                dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # Using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


#####################################################################################################
# Constants
#####################################################################################################
num_examples = 30000
embedding_dim = 256
units = 1024
BATCH_SIZE = 64
EPOCHS = 10

training = ast.literal_eval(sys.argv) if len(sys.argv) > 1 and (
    sys.argv[1] == 'True' or sys.argv[1] == 'False') else False
verbose = False

checkpoint_dir = './training_checkpoints'

#####################################################################################################
# Main
#####################################################################################################

# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

# Save the path
path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# Load the dataset
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
    path_to_file, num_examples)

# Find the max Length for each part
max_length_targ, max_length_inp = max_length(
    target_tensor), max_length(input_tensor)

# Split data into 80/20 mix of training and test data
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    input_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(input_tensor_train)

steps_per_epoch = BATCH_SIZE

vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# Set the Optimizer
optimizer = tf.keras.optimizers.Adam()

# Set up Checkpoints

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_dir, max_to_keep=3)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(checkpoint_manager.latest_checkpoint)
if checkpoint_manager.latest_checkpoint:
    if verbose:
        print("Restoring from {}".format(checkpoint_manager.latest_checkpoint))

if training:
    train()

sentence = input("Please enter a spanish sentence: ")

print(translate(sentence))
