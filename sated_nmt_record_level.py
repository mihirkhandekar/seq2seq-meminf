# -*- coding: utf-8 -*-

import io
import os
import pickle
import re
import time
import unicodedata

import numpy as np
import tensorflow as tf

from models import Decoder, Encoder
from train import Train

path_to_train_en_file = "./sated-release-0.9.0/en-fr/train.en"
path_to_train_fr_file = "./sated-release-0.9.0/en-fr/train.fr"
path_to_valid_en_file = "./sated-release-0.9.0/en-fr/dev.en"
path_to_valid_fr_file = "./sated-release-0.9.0/en-fr/dev.fr"
path_to_test_en_file = "./sated-release-0.9.0/en-fr/test.en"
path_to_test_fr_file = "./sated-release-0.9.0/en-fr/test.fr"

EPOCHS = 20
TO_TRAIN = True
BATCH_SIZE = 128
ATTACKER_KNOWLEDGE_RATIO = 0.5


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sated_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    w = '<start> ' + w + ' <end>'
    return w


def create_sated_dataset(path_lang1, path_lang2, num_examples):
    lines_input_lang = io.open(path_lang1, encoding='UTF-8').read().strip().split('\n')
    lines_target_lang = io.open(path_lang2, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sated_sentence(lang1_line), preprocess_sated_sentence(lang2_line)]
                  for lang1_line, lang2_line in zip(lines_input_lang[:num_examples], lines_target_lang[:num_examples])]
    return zip(*word_pairs)


# en_train, fr_train = create_sated_dataset(path_to_train_en_file, path_to_train_fr_file, 10)
# en_valid, fr_valid = create_sated_dataset(path_to_valid_en_file, path_to_valid_fr_file, 10)
# en_test, fr_test = create_sated_dataset(path_to_test_en_file, path_to_test_fr_file, 10)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_sated_dataset(path_inp_train, path_targ_train, path_inp_test, path_targ_test, num_train=None, num_test=None):
    inp_lang_train, targ_lang_train = create_sated_dataset(path_inp_train, path_targ_train, num_train)
    inp_lang_test, targ_lang_test = create_sated_dataset(path_inp_test, path_targ_test, num_test)

    inp_lang = inp_lang_train + inp_lang_test
    targ_lang = targ_lang_train + targ_lang_test

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


num_train = 3000
num_test = 1000
input_tensor, target_tensor, inp_lang, targ_lang = load_sated_dataset(path_to_train_fr_file,
                                                                      path_to_train_en_file,
                                                                      path_to_test_fr_file,
                                                                      path_to_test_en_file,
                                                                      num_train, num_test)

with open('data/satedrecord/inp_lang.pickle', 'wb') as handle, open('data/satedrecord/targ_lang.pickle', 'wb') as handle2:
    pickle.dump(inp_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(targ_lang, handle2, protocol=pickle.HIGHEST_PROTOCOL)

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

print(f"max_length_targ = {max_length_targ}, max_length_inp = {max_length_inp}")

input_tensor_train = input_tensor[:num_train]
input_tensor_val = input_tensor[num_train:]
target_tensor_train = target_tensor[:num_train]
target_tensor_val = target_tensor[num_train:]

BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE

vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

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

checkpoint_dir = './checkpoints/satedrecord/training_checkpoints'
shadow_checkpoint_dir = './checkpoints/satedrecord/shadow_checkpoints'
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

np.save('data/satedrecord/in_train.npy', in_train)
np.save('data/satedrecord/out_train.npy', out_train)
np.save('data/satedrecord/in_test.npy', in_test)
np.save('data/satedrecord/out_test.npy', out_test)
np.save('data/satedrecord/in_train_label.npy', in_train_label)
np.save('data/satedrecord/out_train_label.npy', out_train_label)
np.save('data/satedrecord/in_test_label.npy', in_test_label)
np.save('data/satedrecord/out_test_label.npy', out_test_label)
