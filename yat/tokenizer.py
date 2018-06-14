from collections import defaultdict
from collections import namedtuple

import MeCab


class Tokenizer():
    def __init__(self):
        self.node = namedtuple("Node", ("surface", "feature"))
        self.token2id = defaultdict(lambda: -1)
        self.id2token = defaultdict(lambda: self.node("UNK", "未知語"))
        self.set_token = set()
        self.num_id = 0

        self._mecab = MeCab.Tagger()
        self._mecab.parse("")

    def tokenize(self, text):
        res = []
        for node in self._mecab.parse(text).split("\n"):
            n = node.split("\t")
            if n == ["EOS"]:
                break
            res.append(self.node(n[0], n[1].split(",")[0]))
        return res

    def fit_on_text(self, text):
        words = self.tokenize(text)

        for w in words:
            if w not in self.set_token:
                self.num_id += 1

                self.token2id[w] = self.num_id
                self.id2token[self.num_id] = w
                self.set_token.add(w)

    def fit_on_texts(self, texts):
        for sentence in texts:
            self.fit_on_text(sentence)

    def text_to_sequence(self, text):
        return [self.token2id[w] for w in self.tokenize(text)]

    def texts_to_sequences(self, texts):
        return [self.text_to_sequence(text) for text in texts]

    def sequence_to_text(self, sequence):
        return "".join([self.id2token[s].surface for s in sequence])

    def sequences_to_texts(self, sequences):
        return ["".join(self.sequence_to_text(sequence)) for sequence in sequences]

    def save_as_text(self, filename):
        with open(filename, "w") as f:
            for w, w_id in self.token2id.items():
                f.write("\t".join([str(w_id), w.surface, w.feature]) + "\n")

    def load_from_text(self, filename):
        with open(filename) as f:
            for line in f:
                l = line.strip().split("\t")
                w_id = int(l[0])
                w = self.node(l[1], l[2])

                self.token2id[w] = w_id
                self.id2token[w_id] = w
                self.num_id += 1
                self.set_token.add(w)
