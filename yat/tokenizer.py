from collections import defaultdict

import MeCab


class Tokenizer():
    def __init__(self):
        self.token2id = defaultdict(lambda: -1)
        self.id2token = defaultdict(lambda: "UNK")
        self.set_token = set()
        self.num_id = 0

        self._wakati = MeCab.Tagger("-Owakati")
        self._wakati.parse("")

    def tokenize(self, text):
        return self._wakati.parse(text).rstrip().split(" ")

    def fit_on_text(self, text):
        words = self.tokenize(text)

        for w in words:
            if w not in self.set_token:
                self.token2id[w] = self.num_id
                self.id2token[self.num_id] = w
                self.num_id += 1
                self.set_token.add(w)

    def fit_on_texts(self, texts):
        [self.fit_on_text(sentence) for sentence in texts]

    def text_to_sequence(self, text):
        return [self.token2id[w] for w in self.tokenize(text)]

    def texts_to_sequences(self, texts):
        return [self.text_to_sequence(text) for text in texts]

    def sequence_to_text(self, sequence):
        return "".join([self.id2token[s] for s in sequence])

    def sequences_to_texts(self, sequences):
        return ["".join(self.sequence_to_text(sequence)) for sequence in sequences]
