import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop


filepath  = tf.keras.utils.get_file('shakespeasre.txt, 'https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa2RZeEpZTDhLTHJSSnppQUV3S2k1WjhMZUdPZ3xBQ3Jtc0ttd1MtTWFwWHpVallZNS1GVzF1LXFYVnZxdDlsUHBwVlhBNHhXT04tdW5fTHkyUmhzVDgzQUQ3c1ZDUEt1Q2ZxNmNlOTZYS003R1BqRHRKTkFWNzlqczE1eGQ5ZVpEOUlhNS1WUmdBMWJha05xYzQxOA&q=https%3A%2F%2Fstorage.googleapis.com%2Fdownload.tensorflow.org%2Fdata%2Fshakespeare.txt&v=QM5XDc4NQJo')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]


characters = sorted(list(set(text)))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_tyo_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3


senteces = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    senteces.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])



X = np.zeros((len(senteces), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(senteces), len(characters)), dtype=np.bool_)


for i, sentence in enumerate(senteces):
    for t, char in enumerate(sentence):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_characters[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
model.fit(X, y, batch_size=256, epochs=4)

model.save('textgerator_model')