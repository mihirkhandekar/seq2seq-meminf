# Approximation Attack 3 : Shadow Models on Scores

import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

SHADOW_MODELS = 4


def get_model(input_dim):
    print('LSTM input dimensions', input_dim)
    model = keras.Sequential()
    model.add(layers.LSTM(256, input_dim=(input_dim,)))
    model.add(layers.Dense(128, kernel_initializer="random_normal",
                           bias_initializer="zeros", activation='relu'))
    model.add(layers.Dense(1, kernel_initializer="random_normal",
                           bias_initializer="zeros", activation='sigmoid'))
    return model

t = np.load('data/train_pred_probs{}.npy'.format(0))
model = get_model(t.shape[-1])
print(model.summary())

y_preds = []
for i in range(SHADOW_MODELS):
    print('Shadow model', i)
    t = np.load('data/train_pred_probs{}.npy'.format(i))
    v = np.load('data/val_pred_probs{}.npy'.format(i))

    print(t.shape, v.shape)

    x = np.concatenate([t, v])
    print(x.shape)

    y = [1. for _ in range(len(t))]
    y.extend([0. for _ in range(len(v))])

    y = np.array(y)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3)

    model.compile(
        loss='binary_crossentropy',
        optimizer="sgd",
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=5
    )

    y_preds.append(model.predict(x_test))

y_pred = np.mean(y_preds, axis=0) > 0.5

print(y_pred.shape, np.array(y_test).shape)

print("Attack 3 Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))
