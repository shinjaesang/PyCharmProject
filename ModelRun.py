import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results

def plot_history(histories, key = 'binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label = name.title() + 'Val')
        plt.plot(history.epoch, history.history[key], color = val[0].get_color(),
                 label = name.title() + 'Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('__','').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.show()

NUM_WORDS = 1000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

checkpoint_path = "baseline_model/cp-0015.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# loss = baseline_model.evaluate(test_data, test_labels, verbose = 2)
# print("훈련되지 않은 모델의 정확도 : {:5.2f}%".format(100*loss[1]))

baseline_model.load_weights(checkpoint_path)
loss = baseline_model.evaluate(test_data, test_labels, verbose=2)
print("복원된 모델의 정확도 : {:5.2f}%".format(100*loss[1]))