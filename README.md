# yat: Yet Another Tokenizer for Japanese NLP

## 使い方

### MeCabを使った分かち書き

```py
import yat

tokenizer = yat.Tokenizer()
tokenizer.tokenize("犬を飼っています")
# ['犬', 'を', '飼っ', 'て', 'い', 'ます']
```

### 単語からIDに変換

```py
import yat

tokenizer = yat.Tokenizer()

text = ["犬を飼っています", "猫を飼っています"]
tokenizer.fit_on_texts(text)
tokenizer.texts_to_sequences(text)
# [[0, 1, 2, 3, 4, 5], [6, 1, 2, 3, 4, 5]]
```
