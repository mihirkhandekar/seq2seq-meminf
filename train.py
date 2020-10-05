import re
import unicodedata

import numpy as np
import tensorflow as tf


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    w = '<start> ' + w + ' <end>'
    return w

class Translate:
    def __init__(self, encoder, decoder, units, inp_lang, targ_lang, max_length_targ, max_length_inp) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.units = units
        self.inp_lang = inp_lang
        self.targ_lang = targ_lang
        self.max_length_targ = max_length_targ
        self.max_length_inp = max_length_inp


    def translate(self, sentence, tensor=False):
        if tensor:
            sen = ''
            for word in sentence:
                if word == 0:
                    break
                sen += self.inp_lang.index_word[word] + ' '
            sentence = sen.split(' ', 1)[1]
            sentence = sentence.rsplit(' ', 1)[0].rsplit(' ', 1)[0]
        result, sentence, attention_plot, pred_probs = self.evaluate(
            sentence, tensor)
        return result, pred_probs


    def evaluate(self, sentence, tensor=False):
        attention_plot = np.zeros((self.max_length_targ, self.max_length_inp))
        sentence = preprocess_sentence(sentence)
        inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen=self.max_length_inp,
                                                            padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']], 0)

        pred_probs = []

        for t in range(self.max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                dec_hidden,
                                                                enc_out)

            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()
            
            predicted_id = tf.argmax(predictions[0]).numpy()
            pred_probs.append(predictions[0].numpy())
            
            if predicted_id:
                result += self.targ_lang.index_word[predicted_id] + ' '
            
            # finish prediction if <end> token is predicted
            # don't perform check if 0 is predicted as it's reserved for padding (word index won't have key = 0)
            if predicted_id != 0 and self.targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot, pred_probs
            
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot, pred_probs


class Train:
    def __init__(self, encoder, decoder, optimizer, loss_function, batch_size, targ_lang):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.targ_lang = targ_lang

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims(
                [self.targ_lang.word_index['<start>']] * self.batch_size, 1)
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_output)

                loss += self.loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss
