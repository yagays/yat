# yat: Yet Another Tokenizer for Japanese NLP

## 使い方

### MeCabを使った分かち書き

```py
import yat

tokenizer = yat.Tokenizer()
sequence = tokenizer.tokenize("犬を飼っています")
```

```py
In []: sequence
Out[]: ['犬', 'を', '飼っ', 'て', 'い', 'ます']
```

### 単語からIDに変換

```py
import yat

tokenizer = yat.Tokenizer()

text = ["犬を飼っています", "猫を飼っています"]
tokenizer.fit_on_texts(text)
sequence = tokenizer.texts_to_sequences(text)
```

```py
In []: sequence
Out[]: [[0, 1, 2, 3, 4, 5], [6, 1, 2, 3, 4, 5]]
```

### Kerasとの連携

```py
import yat
from keras import preprocessing

texts = ["吾輩は猫である。名前はまだ無い。",
         "どこで生れたかとんと見当がつかぬ。",
         "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"]
tokenizer = yat.Tokenizer()
tokenizer.fit_on_texts(texts)
sequence = tokenizer.texts_to_sequences(texts)

maxlen = 20
X_train = preprocessing.sequence.pad_sequences(sequence, maxlen=20)
```

```py
In []: X_train
Out[]:
array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6, 1,  7,  8,  5],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  3, 10, 11, 12, 13, 14, 15, 16, 17,  5],
       [ 0, 18, 19, 20, 21, 22, 11, 23,  3, 24, 25, 26, 27, 28,  1, 29, 22, 26, 30,  5]], dtype=int32)
```
