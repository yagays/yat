import yat

import numpy as np
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

texts_neko = ["吾輩は猫である。名前はまだ無い。",
              "どこで生れたかとんと見当がつかぬ。",
              "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"]

texts_melos = ["メロスは激怒した。",
               "必ず、かの邪智暴虐の王を除かなければならぬと決意した。",
               "メロスには政治がわからぬ。"]

# tokenize by using yat
tokenizer = yat.Tokenizer()
tokenizer.fit_on_texts(texts_neko + texts_melos)
sequence_neko = tokenizer.texts_to_sequences(texts_neko)
sequence_melos = tokenizer.texts_to_sequences(texts_melos)


# The same as Keras Preprocessing
maxlen = 20
X_train_neko = preprocessing.sequence.pad_sequences(sequence_neko, maxlen=maxlen)
X_train_melos = preprocessing.sequence.pad_sequences(sequence_melos, maxlen=maxlen)
X_train = np.r_[X_train_neko, X_train_melos]

y_train = np.array([0, 0, 0, 1, 1, 1])

# train
model = Sequential()
model.add(Dense(20, input_shape=(maxlen,)))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=1, epochs=1, shuffle=True)


# test
sequence_test = tokenizer.texts_to_sequences(["メロスは、村の牧人である。"])
# [[31, 1, 34, -1, 38, -1, 3, 4, 5]]
X_test = preprocessing.sequence.pad_sequences(sequence_test, maxlen=maxlen)
# array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 31,  1, 34, -1, 38, -1,  3,  4,  5]], dtype=int32)
model.predict(X_test)
# array([[0.9999999]], dtype=float32)
