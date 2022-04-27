# Approximation Attack 2 : Shadow Models on Rank

import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from shadow_model import SHADOW_UNITS, ShadowDecoder, ShadowEncoder
from train import Train, Translate

NUM_SHADOW_MODELS = 4
BATCH_SIZE = 128
EPOCHS = 15
shadow_checkpoint_dir = './checkpoints/satedrecord/shadow_checkpoints'

with open('data/satedrecord/inp_lang.pickle', 'rb') as handle, open('data/satedrecord/targ_lang.pickle', 'rb') as handle2:
    inp_lang = pickle.load(handle)
    targ_lang = pickle.load(handle2)


in_train, in_train_label = np.load(
    'data/satedrecord/in_train.npy'), np.load('data/satedrecord/in_train_label.npy')
out_train, out_train_label = np.load(
    'data/satedrecord/out_train.npy'), np.load('data/satedrecord/out_train_label.npy')
in_test, in_test_label = np.load(
    'data/satedrecord/in_test.npy'), np.load('data/satedrecord/in_test_label.npy')
out_test, out_test_label = np.load(
    'data/satedrecord/out_test.npy'), np.load('data/satedrecord/out_test_label.npy')

print(in_train.shape, in_train_label.shape,
      out_train.shape, out_train_label.shape)

BUFFER_SIZE = len(in_train)

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_length_targ, max_length_inp = 65, 67

minimum = min(len(in_train), len(out_train))


ds_size = minimum // NUM_SHADOW_MODELS

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def translate_and_get_indices(tr, tar, pred_probs):
    res = ''.join(f'{targ_lang.index_word[word]} ' for word in tar if word != 0)
    res = res.split(' ', 1)[1]

    ### score = sentence_bleu([tr.split()], res.split())

    indices = []

    for word, prob in zip(res.split(), pred_probs):
        temp = (-prob).argsort()[:len(prob)]
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(prob))
        ind = targ_lang.word_index[word]
        indices.append(ranks[ind])

    return indices


train_indices = []
test_indices = []

for m in range(NUM_SHADOW_MODELS):
    # TODO : Change inp_size
    print('Training shadow model', m)
    input_tensor_train_slice = in_train[m * ds_size: (m+1) * ds_size]
    target_tensor_train_slice = in_train_label[m *
                                               ds_size: (m+1) * ds_size]

    input_tensor_val_slice = out_train[m * ds_size: (m+1) * ds_size]
    target_tensor_val_slice = out_train_label[m * ds_size: (m+1) * ds_size]

    shadow_encoder = ShadowEncoder(vocab_inp_size, BATCH_SIZE)
    shadow_decoder = ShadowDecoder(vocab_tar_size, BATCH_SIZE)

    shadow_optimizer = tf.keras.optimizers.Adam()

    shadow_checkpoint_prefix = os.path.join(
        shadow_checkpoint_dir + str(m), f"ckptshadow{str(m)}"
    )


    shadow_checkpoint = tf.train.Checkpoint(optimizer=shadow_optimizer,
                                            encoder=shadow_encoder,
                                            decoder=shadow_decoder)

    dataset = tf.data.Dataset.from_tensor_slices(
        (in_train, in_train_label)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    train = Train(shadow_encoder, shadow_decoder, shadow_optimizer,
                  loss_function, BATCH_SIZE, targ_lang)
    steps_per_epoch = len(in_train)//BATCH_SIZE

    translator = Translate(shadow_encoder, shadow_decoder, SHADOW_UNITS,
                                inp_lang, targ_lang, max_length_targ, max_length_inp)

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = shadow_encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train.train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))

        if (epoch + 1) % 2 == 0:
            shadow_checkpoint.save(file_prefix=shadow_checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    in_train_indices = []
    pred_probs_in_train = []

    i = 0
    for ten, tar in zip(in_train, in_train_label):
        i += 1
        if i > ds_size:
            break
        tr, pred_probs = translator.translate(ten, True)
        pred_probs_in_train.append(pred_probs)
        indices = translate_and_get_indices(tr, tar, pred_probs)
        in_train_indices.append(np.mean(indices))

    out_train_indices = []
    pred_probs_out_train = []
    i = 0
    for ten, tar in zip(out_train, out_train_label):
        i += 1
        if i > ds_size:
            break
        tr, pred_probs = translator.translate(ten, True)
        pred_probs_out_train.append(pred_probs)
        indices = translate_and_get_indices(tr, tar, pred_probs)
        out_train_indices.append(np.mean(indices))

    train_indices.append((in_train_indices, out_train_indices))

    in_test_indices = []
    pred_probs_in_test = []
    i = 0
    for ten, tar in zip(in_test, in_test_label):
        i += 1
        if i > ds_size:
            break
        tr, pred_probs = translator.translate(ten, True)
        pred_probs_in_test.append(pred_probs)
        indices = translate_and_get_indices(tr, tar, pred_probs)
        in_test_indices.append(np.mean(indices))

    out_test_indices = []
    pred_probs_out_test = []
    i = 0
    for ten, tar in zip(out_test, out_test_label):
        i += 1
        if i > ds_size:
            break
        tr, pred_probs = translator.translate(ten, True)
        pred_probs_out_test.append(pred_probs)
        indices = translate_and_get_indices(tr, tar, pred_probs)
        out_test_indices.append(np.mean(indices))

    test_indices.append((in_test_indices, out_test_indices))


################################################################
y_preds = []
classifiers = []
for i in range(NUM_SHADOW_MODELS):
    t, v = train_indices[i]
    print('Shadow model', i)
    x_train = np.concatenate([t, v])
    y_train = [1. for _ in range(len(t))]
    y_train.extend([0. for _ in range(len(v))])

    clf = svm.SVC()
    clf.fit(x_train.reshape(-1, 1), y_train)
    classifiers.append(clf)
    x_test = np.concatenate(test_indices[i])
    test_size = len(test_indices[i][0])
    y_test = [1. for _ in range(test_size)]
    y_test.extend([0. for _ in range(test_size)])

    y_pred = clf.predict(x_test.reshape(-1, 1))
    y_preds.append(y_pred)


y_pred = np.mean(y_preds, axis=0) > 0.5

print(y_pred.shape, np.array(y_test).shape)

print("Attack 2 Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))

ra_score = roc_auc_score(y_test, y_pred)
print("Attack 2 ROC_AUC Score : %.2f%%" % (100.0 * ra_score))

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
plt.figure(1)
plt.plot(fpr, tpr, label='Attack 2')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig('satedrecord_attack2_roc_curve.png')
