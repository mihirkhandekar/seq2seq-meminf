# -*- coding: utf-8 -*-

import io
import os
import pickle
import re
import time
import unicodedata

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models import Decoder, Encoder
from train import Train

path_to_file = "./spa-eng/spa.txt"

EPOCHS = 20
TO_TRAIN = True
BATCH_SIZE = 128
ATTACKER_KNOWLEDGE_RATIO = 0.5


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    w = f'<start> {w} <end>'
    return w


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split(
        '\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


en, sp = create_dataset(path_to_file, None)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
    path_to_file, num_examples)

with open('data/inp_lang.pickle', 'wb') as handle, open('data/targ_lang.pickle', 'wb') as handle2:
    pickle.dump(inp_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(targ_lang, handle2, protocol=pickle.HIGHEST_PROTOCOL)

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    input_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


encoder = Encoder(vocab_inp_size, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, BATCH_SIZE)

checkpoint_dir = './checkpoints/training_checkpoints'
shadow_checkpoint_dir = './checkpoints/shadow_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


if TO_TRAIN:  # If train
    train = Train(encoder, decoder, optimizer,
                  loss_function, BATCH_SIZE, targ_lang)
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train.train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        if (epoch + 1) % 2 == 0:
            print('Saving model')
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


minimum = min(len(input_tensor_train), len(input_tensor_val))

in_train = input_tensor_train[: int(ATTACKER_KNOWLEDGE_RATIO * minimum)]
in_train_label = target_tensor_train[: int(ATTACKER_KNOWLEDGE_RATIO * minimum)]
out_train = input_tensor_val[: int(ATTACKER_KNOWLEDGE_RATIO * minimum)]
out_train_label = target_tensor_val[: int(ATTACKER_KNOWLEDGE_RATIO * minimum)]
in_test = input_tensor_train[int(ATTACKER_KNOWLEDGE_RATIO * minimum):]
in_test_label = target_tensor_train[int(ATTACKER_KNOWLEDGE_RATIO * minimum):]
out_test = input_tensor_val[int(ATTACKER_KNOWLEDGE_RATIO * minimum):]
out_test_label = target_tensor_val[int(ATTACKER_KNOWLEDGE_RATIO * minimum):]

np.save('data/in_train.npy', in_train)
np.save('data/out_train.npy', out_train)
np.save('data/in_test.npy', in_test)
np.save('data/out_test.npy', out_test)
np.save('data/in_train_label.npy', in_train_label)
np.save('data/out_train_label.npy', out_train_label)
np.save('data/in_test_label.npy', in_test_label)
np.save('data/out_test_label.npy', out_test_label)
