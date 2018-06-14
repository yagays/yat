from collections import defaultdict
from collections import namedtuple

import MeCab


class Tokenizer():
    def __init__(self, mecab_param=""):
        self.node = namedtuple("Node", ("surface", "feature"))
        self.token2id = defaultdict(lambda: -1)
        self.id2token = defaultdict(lambda: self.node("UNK", "未知語"))
        self.set_token = set()
        self.word_counts = defaultdict(lambda: 0)
        self.word_index = 0

        self.filtering = defaultdict(set)

        w_blank = self.node("", "")
        self.token2id[w_blank] = self.word_index
        self.id2token[self.word_index] = w_blank
        self.set_token.add(w_blank)

        self._mecab = MeCab.Tagger(mecab_param)
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
            self.word_counts[w] += 1

            if w not in self.set_token:
                self.word_index += 1

                self.token2id[w] = self.word_index
                self.id2token[self.word_index] = w
                self.set_token.add(w)

    def fit_on_texts(self, texts):
        for sentence in texts:
            self.fit_on_text(sentence)

    def text_to_sequence(self, text):
        res = []
        for w in self.tokenize(text):
            if self.filter_text(w):
                res.append(self.token2id[w])
        return res

    def texts_to_sequences(self, texts):
        return [self.text_to_sequence(text) for text in texts]

    def sequence_to_text(self, sequence):
        return "".join([self.id2token[s].surface for s in sequence])

    def sequences_to_texts(self, sequences):
        return ["".join(self.sequence_to_text(sequence)) for sequence in sequences]

    def filter_text(self, node):
        ret = True
        if "vocabulary_size" in self.filtering and node not in self.filtering["vocabulary_size"]:
            ret = False

        if "pos" in self.filtering and node.feature not in self.filtering["pos"]:
            ret = False

        return ret

    def filter_by_vocabulary_size(self, n):
        top_n_counts = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        self.filtering["vocabulary_size"] = set(s[0] for s in top_n_counts)

    def filter_by_pos(self, pos_list):
        if pos_list:
            self.filtering["pos"] = set(pos_list)

    def save_as_text(self, filename, filter=False):
        with open(filename, "w") as f:
            for w, w_id in self.token2id.items():
                if filter and not self.filter_text(w):
                    continue

                f.write("\t".join([str(w_id), w.surface, w.feature]) + "\n")

    def load_from_text(self, filename):
        with open(filename) as f:
            for line in f:
                l = line.strip().split("\t")
                w_id = int(l[0])
                w = self.node(l[1], l[2])

                self.token2id[w] = w_id
                self.id2token[w_id] = w
                self.word_index += 1
                self.set_token.add(w)
