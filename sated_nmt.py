from collections import defaultdict

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense, Add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2

from load_sated import load_sated_data_by_user
from helper import DenseTransposeTied, Attention

MODEL_PATH = 'checkpoints/sated/model/'
OUTPUT_PATH = 'checkpoints/sated/output/'

tf.compat.v1.disable_eager_execution()


def group_texts_by_len(src_texts, trg_texts, bs=20):
    print("Bucketing batches")
    # Bucket samples by source sentence length
    buckets = defaultdict(list)
    batches = []
    for src, trg in zip(src_texts, trg_texts):
        buckets[len(src)].append((src, trg))

    for src_len, bucket in buckets.items():
        np.random.shuffle(bucket)
        num_batches = int(np.ceil(len(bucket) * 1.0 / bs))
        for i in range(num_batches):
            cur_batch_size = bs if i < num_batches - 1 else len(bucket) - bs * i
            batches.append(([bucket[i * bs + j][0] for j in range(cur_batch_size)],
                            [bucket[i * bs + j][1] for j in range(cur_batch_size)]))
    return batches


def build_nmt_model(Vs, Vt, demb=128, h=128, drop_p=0.5, tied=True, mask=True, attn=True, l2_ratio=1e-4,
                    training=None, rnn_fn='lstm'):
    if rnn_fn in ['lstm', 'gru']:
        rnn = LSTM
    else:
        raise ValueError(rnn_fn)

    # build encoder
    encoder_input = Input((None,), dtype='float32', name='encoder_input')
    if mask:
        encoder_emb_layer = Embedding(Vs + 1, demb, mask_zero=True, embeddings_regularizer=l2(l2_ratio),
                                      name='encoder_emb')
    else:
        encoder_emb_layer = Embedding(Vs, demb, mask_zero=False, embeddings_regularizer=l2(l2_ratio),
                                      name='encoder_emb')

    encoder_emb = encoder_emb_layer(encoder_input)

    if drop_p > 0.:
        encoder_emb = Dropout(drop_p)(encoder_emb, training=training)

    encoder_rnn = rnn(h, return_sequences=True, return_state=True, kernel_regularizer=l2(l2_ratio), name='encoder_rnn')
    encoder_rtn = encoder_rnn(encoder_emb)
    # encoder_outputs, encoder_h, encoder_c = encoder_rnn(encoder_emb)
    encoder_outputs = encoder_rtn[0]
    encoder_states = encoder_rtn[1:]

    # build decoder
    decoder_input = Input((None,), dtype='float32', name='decoder_input')
    if mask:
        decoder_emb_layer = Embedding(Vt + 1, demb, mask_zero=True, embeddings_regularizer=l2(l2_ratio),
                                      name='decoder_emb')
    else:
        decoder_emb_layer = Embedding(Vt, demb, mask_zero=False, embeddings_regularizer=l2(l2_ratio),
                                      name='decoder_emb')

    decoder_emb = decoder_emb_layer(decoder_input)

    if drop_p > 0.:
        decoder_emb = Dropout(drop_p)(decoder_emb, training=training)

    decoder_rnn = rnn(h, return_sequences=True, kernel_regularizer=l2(l2_ratio), name='decoder_rnn')
    decoder_outputs = decoder_rnn(decoder_emb, initial_state=encoder_states)

    if drop_p > 0.:
        decoder_outputs = Dropout(drop_p)(decoder_outputs, training=training)

    if tied:
        final_outputs = DenseTransposeTied(Vt, kernel_regularizer=l2(l2_ratio), name='outputs',
                                           tied_to=decoder_emb_layer, activation='linear')(decoder_outputs)
    else:
        final_outputs = Dense(Vt, activation='linear', kernel_regularizer=l2(l2_ratio), name='outputs')(decoder_outputs)

    if attn:
        contexts = Attention(units=h, kernel_regularizer=l2(l2_ratio), name='attention',
                             use_bias=False)([encoder_outputs, decoder_outputs])
        if drop_p > 0.:
            contexts = Dropout(drop_p)(contexts, training=training)

        contexts_outputs = Dense(Vt, activation='linear', use_bias=False, name='context_outputs',
                                 kernel_regularizer=l2(l2_ratio))(contexts)

        final_outputs = Add(name='final_outputs')([final_outputs, contexts_outputs])

    return Model(inputs=[encoder_input, decoder_input], outputs=[final_outputs])


def build_inference_decoder(mask=False, demb=128, h=128, Vt=5000, tied=True, attn=True):
    rnn = LSTM

    # build decoder
    decoder_input = Input(batch_shape=(None, None), dtype='float32', name='decoder_input')
    encoder_outputs = Input(batch_shape=(None, None, h), dtype='float32', name='encoder_outputs')
    encoder_h = Input(batch_shape=(None, h), dtype='float32', name='encoder_h')
    encoder_c = Input(batch_shape=(None, h), dtype='float32', name='encoder_c')

    if mask:
        decoder_emb_layer = Embedding(Vt + 1, demb, mask_zero=True,
                                      name='decoder_emb')
    else:
        decoder_emb_layer = Embedding(Vt, demb, mask_zero=False,
                                      name='decoder_emb')

    decoder_emb = decoder_emb_layer(decoder_input)

    decoder_rnn = rnn(h, return_sequences=True, name='decoder_rnn')
    decoder_outputs = decoder_rnn(decoder_emb, initial_state=[encoder_h, encoder_c])

    if tied:
        final_outputs = DenseTransposeTied(Vt, name='outputs',
                                           tied_to=decoder_emb_layer, activation='linear')(decoder_outputs)
    else:
        final_outputs = Dense(Vt, activation='linear', name='outputs')(decoder_outputs)

    if attn:
        contexts = Attention(units=h, use_bias=False, name='attention')([encoder_outputs, decoder_outputs])
        contexts_outputs = Dense(Vt, activation='linear', use_bias=False, name='context_outputs')(contexts)
        final_outputs = Add(name='final_outputs')([final_outputs, contexts_outputs])

    inputs = [decoder_input, encoder_outputs, encoder_h, encoder_c]
    model = Model(inputs=inputs, outputs=[final_outputs])
    return model


def words_to_indices(data, vocab, mask=True):
    if mask:
        return [[vocab[w] + 1 for w in t] for t in data]
    else:
        return [[vocab[w] for w in t] for t in data]


def pad_texts(texts, eos, mask=True):
    maxlen = max(len(t) for t in texts)
    for t in texts:
        while len(t) < maxlen:
            if mask:
                t.insert(0, 0)
            else:
                t.append(eos)
    return np.asarray(texts, dtype='float32')


def get_perp(user_src_data, user_trg_data, pred_fn, prop=1.0, shuffle=False):
    loss = 0.
    iters = 0.

    indices = np.arange(len(user_src_data))
    n = int(prop * len(indices))

    if shuffle:
        np.random.shuffle(indices)

    for idx in indices[:n]:
        src_text = np.asarray(user_src_data[idx], dtype=np.float32).reshape(1, -1)
        trg_text = np.asarray(user_trg_data[idx], dtype=np.float32)
        trg_input = trg_text[:-1].reshape(1, -1)
        trg_label = trg_text[1:].reshape(1, -1)

        err = pred_fn([src_text, trg_input, trg_label, 0])[0]

        loss += err
        iters += trg_label.shape[1]

    return loss, iters


def train_sated_nmt(loo=0, num_users=200, num_words=5000, num_epochs=20, h=128, emb_h=128, l2_ratio=1e-4, exp_id=0,
                    lr=0.001, batch_size=32, mask=False, drop_p=0.5, cross_domain=False, tied=False, ablation=False,
                    sample_user=False, user_data_ratio=0., rnn_fn='lstm', optim_fn='adam'):
    # if cross_domain:
    #     sample_user = True
    #     user_src_texts, user_trg_texts, dev_src_texts, dev_trg_texts, test_src_texts, test_trg_texts,\
    #         src_vocabs, trg_vocabs = load_europarl_by_user(num_users=num_users, num_words=num_words)
    # else:
    user_src_texts, user_trg_texts, dev_src_texts, dev_trg_texts, test_src_texts, test_trg_texts, \
    src_vocabs, trg_vocabs = load_sated_data_by_user(num_users, num_words, sample_user=sample_user,
                                                     user_data_ratio=user_data_ratio)
    train_src_texts, train_trg_texts = [], []

    users = sorted(user_src_texts.keys())

    print("Creating dataset...")
    for i, user in enumerate(users):
        if loo is not None and i == loo:
            print(f"Leave user {user} out")
            continue
        train_src_texts += user_src_texts[user]
        train_trg_texts += user_trg_texts[user]

    train_src_texts = words_to_indices(train_src_texts, src_vocabs, mask=mask)
    train_trg_texts = words_to_indices(train_trg_texts, trg_vocabs, mask=mask)
    dev_src_texts = words_to_indices(dev_src_texts, src_vocabs, mask=mask)
    dev_trg_texts = words_to_indices(dev_trg_texts, trg_vocabs, mask=mask)

    print(
        f"Num train data {len(train_src_texts)}, num test data {len(dev_src_texts)}"
    )


    Vs = len(src_vocabs)
    Vt = len(trg_vocabs)
    print(f"Source vocab len: {Vs}, Target vocab length: {Vt}")

    print("Building NMT model...")
    model = build_nmt_model(Vs=Vs, Vt=Vt, mask=mask, drop_p=drop_p, h=h, demb=emb_h, tied=tied, l2_ratio=l2_ratio,
                            rnn_fn=rnn_fn)
    src_input_var, trg_input_var = model.inputs
    prediction = model.output

    trg_label_var = K.placeholder((None, None), dtype='float32')

    loss = K.sparse_categorical_crossentropy(trg_label_var, prediction, from_logits=True)
    loss = K.mean(K.sum(loss, axis=-1))

    if optim_fn == 'adam':
        optimizer = Adam(learning_rate=lr, clipnorm=5.)
    elif optim_fn == 'mom_sgd':
        optimizer = SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError(optim_fn)

    updates = optimizer.get_updates(loss, model.trainable_weights)

    train_fn = K.function(inputs=[src_input_var, trg_input_var, trg_label_var, K.learning_phase()], outputs=[loss],
                          updates=updates)
    pred_fn = K.function(inputs=[src_input_var, trg_input_var, trg_label_var, K.learning_phase()], outputs=[loss])

    # pad batches to same length
    train_prop = 0.2
    batches = []
    for batch in group_texts_by_len(train_src_texts, train_trg_texts, bs=batch_size):
        src_input, trg_input = batch
        src_input = pad_texts(src_input, src_vocabs['<eos>'], mask=mask)
        trg_input = pad_texts(trg_input, trg_vocabs['<eos>'], mask=mask)
        batches.append((src_input, trg_input))

    print(f"Number of batches: {len(batches)}\nFirst batch: {batches[0]}")

    print("Training NMT model...")
    for epoch in range(num_epochs):
        print(f"On epoch {epoch} of training...")
        np.random.shuffle(batches)

        for batch in batches:
            src_input, trg_input = batch
            _ = train_fn([src_input, trg_input[:, :-1], trg_input[:, 1:], 1])[0]

        train_loss, train_it = get_perp(train_src_texts, train_trg_texts, pred_fn, shuffle=True, prop=train_prop)
        test_loss, test_it = get_perp(dev_src_texts, dev_trg_texts, pred_fn)

        print("Epoch {}, train loss={:.3f}, train perp={:.3f}, test loss={:.3f}, test perp={:.3f}".format(
            epoch,
            train_loss / len(train_src_texts) / train_prop,
            np.exp(train_loss / train_it),
            test_loss / len(dev_src_texts),
            np.exp(test_loss / test_it)))

    # if cross_domain:
    #     fname = 'europal_nmt{}'.format('' if loo is None else loo)
    # else:
    fname = f"sated_nmt{'' if loo is None else loo}"

    if ablation:
        fname = f'ablation_{fname}'

    if 0. < user_data_ratio < 1.:
        fname += f'_dr{user_data_ratio}'

    if sample_user:
        fname += f'_shadow_exp{exp_id}_{rnn_fn}'
        np.savez(
            MODEL_PATH
            + f"shadow_users{exp_id}_{rnn_fn}_{num_users}_{'cd' if cross_domain else ''}.npz",
            users,
        )

        print(
            f"""Shadow model {exp_id} saved to {MODEL_PATH + f"shadow_users{exp_id}_{rnn_fn}_{num_users}_{'cd' if cross_domain else ''}.npz"}."""
        )


    model.save(MODEL_PATH + f'{fname}_{num_users}.h5')
    print(f"Target model saved to {MODEL_PATH + f'{fname}_{num_users}.h5'}.")
    K.clear_session()


if __name__ == '__main__':
    # Following reproducibility info in the auditing seq2seq models paper (https://arxiv.org/pdf/1811.00513.pdf)
    # Define hyperparameters for target model
    epochs = 30
    batch_size = 20
    lr = 0.001
    rnn_fn = 'lstm'
    optim_fn = 'adam'
    num_users = 300
    sample_user_flag = False
    cross_domain_flag = False

    print("Get target model...")
    train_sated_nmt(exp_id=None, loo=None, sample_user=sample_user_flag,
                    lr=lr, cross_domain=cross_domain_flag, h=128, emb_h=128,
                    num_epochs=epochs, num_users=num_users, batch_size=batch_size,
                    drop_p=0.5, rnn_fn=rnn_fn, optim_fn=optim_fn)

    # Update hyperparameters for shadow models
    num_shadow_models = 10
    epochs = 50
    batch_size = 35
    lr = 0.01
    rnn_fn = 'gru'
    optim_fn = 'mom_sgd'
    sample_user_flag = True
    dims = list(range(64, 353, 32))

    print("Get shadow models...")
    for i in range(num_shadow_models):
        print(f"Setting hidden/embedding dimension = {dims[i]} for shadow model {i}")
        train_sated_nmt(exp_id=i, loo=None, sample_user=sample_user_flag,
                        lr=lr, cross_domain=cross_domain_flag, h=dims[i], emb_h=dims[i],
                        num_epochs=epochs, num_users=num_users, batch_size=batch_size,
                        drop_p=0, rnn_fn=rnn_fn, optim_fn=optim_fn)
