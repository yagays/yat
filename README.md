# yat: Yet Another Tokenizer for Japanese NLP

## インストール

```sh
$ pip install git+https://github.com/yagays/yat
```

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

### 文章からIDに変換

```py
import yat

tokenizer = yat.Tokenizer()

text = ["犬を飼っています", "猫を飼っています"]
tokenizer.fit_on_texts(text)
```

```py
In []: tokenizer.texts_to_sequences(text)
Out[]: [[0, 1, 2, 3, 4, 5], [6, 1, 2, 3, 4, 5]]
```

### IDから文章に変換

```py
In []: tokenizer.sequences_to_texts([[0, 1, 2, 3, 4, 5], [6, 1, 2, 3, 4, 5]])
Out[]: ['犬を飼っています', '猫を飼っています']
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
X_train = preprocessing.sequence.pad_sequences(sequence, maxlen=maxlen)
```

```py
In []: X_train
Out[]:
array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6, 1,  7,  8,  5],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  3, 10, 11, 12, 13, 14, 15, 16, 17,  5],
       [ 0, 18, 19, 20, 21, 22, 11, 23,  3, 24, 25, 26, 27, 28,  1, 29, 22, 26, 30,  5]], dtype=int32)
```

### Tokenizerの書き出し

```py
tokenizer.save_as_text("yat_tokenizer.txt")
```

```sh
$ head yat_tokenizer.txt
1       吾輩    名詞
2       は      助詞
3       猫      名詞
4       で      助動詞
5       ある    助動詞
```

### Tokenizerの読み込み

```py
tokenizer = yat.Tokenizer()
tokenizer.load_from_text("yat_tokenizer.txt")
```

```py
In []: tokenizer.token2id
Out[]:
defaultdict(<function yat.tokenizer.Tokenizer.__init__.<locals>.<lambda>()>,
            {Node(surface='吾輩', feature='名詞'): 1,
             Node(surface='は', feature='助詞'): 2,
             Node(surface='猫', feature='名詞'): 3,
             Node(surface='で', feature='助動詞'): 4,
             Node(surface='ある', feature='助動詞'): 5,
[...]
```

### Neologdを利用する

```py
tokenizer = Tokenizer("-d /path/to/mecab-ipadic-neologd")
```
