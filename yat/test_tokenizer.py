import os

import pytest
from yat import Tokenizer


def test_tokenizer():
    text = ["犬を飼っています", "猫を飼っています"]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    assert tokenizer.word_index == 7
    assert tokenizer.texts_to_sequences(text) == [[1, 2, 3, 4, 5, 6], [7, 2, 3, 4, 5, 6]]
    assert tokenizer.sequences_to_texts([[1, 2, 3, 4, 5, 6], [7, 2, 3, 4, 5, 6]]) == text

    # unknown word: "狼"
    assert tokenizer.text_to_sequence("狼を飼っています") == [-1, 2, 3, 4, 5, 6]
    assert tokenizer.sequence_to_text([-1, 2, 3, 4, 5, 6]) == "UNKを飼っています"

    # fit additional word "狼"
    tokenizer.fit_on_text("狼を飼っています")
    assert tokenizer.word_index == 8
    assert tokenizer.text_to_sequence("狼を飼っています") == [8, 2, 3, 4, 5, 6]

    tokenizer.fit_on_texts(["狐を飼っています", "狸を飼っています"])
    assert tokenizer.word_index == 10
    assert tokenizer.texts_to_sequences(["狐を飼っています", "狸を飼っています"]) == [[9, 2, 3, 4, 5, 6], [10, 2, 3, 4, 5, 6]]

    # for padding
    assert tokenizer.sequence_to_text([0, 0, -1, 2, 3, 4, 5, 6]) == "UNKを飼っています"


def test_tokenizer_blank():
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(["", "", ""])
    assert len(tokenizer.set_token) == 1


def test_tokenizer_filtering_vocabulary_size():
    text = "すもももももももものうち"
    tokenizer = Tokenizer()
    tokenizer.fit_on_text(text)

    # before filtering
    assert len(tokenizer.set_token) == 6
    assert len(tokenizer.filtering["vocabulary_size"]) == 0

    # after filtering
    tokenizer.filter_by_vocabulary_size(2)
    assert len(tokenizer.filtering["vocabulary_size"]) == 2
    assert tokenizer.text_to_sequence(text) == [2, 3, 2, 3]  # すもも も もも も もも の うち
    assert tokenizer.sequence_to_text([2, 3, 2, 3]) == "もももももも"


def test_tokenizer_filtering_vocabulary_size():
    text = "すもももももももものうち"
    tokenizer = Tokenizer()
    tokenizer.fit_on_text(text)

    # すもももももももものうち
    # すもも  名詞,一般,*,*,*,*,すもも,スモモ,スモモ
    # も      助詞,係助詞,*,*,*,*,も,モ,モ
    # もも    名詞,一般,*,*,*,*,もも,モモ,モモ
    # も      助詞,係助詞,*,*,*,*,も,モ,モ
    # もも    名詞,一般,*,*,*,*,もも,モモ,モモ
    # の      助詞,連体化,*,*,*,*,の,ノ,ノ
    # うち    名詞,非自立,副詞可能,*,*,*,うち,ウチ,ウチ

    tokenizer.filter_by_pos(["名詞"])
    assert len(tokenizer.filtering["pos"]) == 1
    assert tokenizer.text_to_sequence(text) == [1, 3, 3, 5]
    assert tokenizer.sequence_to_text([1, 3, 3, 5]) == "すももももももうち"
    assert tokenizer.sequence_to_text([1, 2, 3, 2, 3, 4, 5]) == text

    tokenizer.filter_by_pos(["名詞", "助詞"])
    assert len(tokenizer.filtering["pos"]) == 2
    assert tokenizer.text_to_sequence(text) == [1, 2, 3, 2, 3, 4, 5]
    assert tokenizer.sequence_to_text([1, 2, 3, 2, 3, 4, 5]) == text


def test_tokenizer_io():
    text = ["犬を飼っています", "猫を飼っています"]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    # prepare
    tmp_dir = ".pytest_cache/"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    output_file = os.path.join(tmp_dir, "test_tokenizer.txt")
    assert tokenizer.text_to_sequence("猫を飼っています") == [7, 2, 3, 4, 5, 6]
    assert tokenizer.sequence_to_text([7, 2, 3, 4, 5, 6]) == "猫を飼っています"

    # save
    tokenizer.save_as_text(output_file)
    with open(output_file) as f:
        num_lines = len(f.readlines())
    assert num_lines == 7

    # load
    tokenizer_load = Tokenizer()
    tokenizer_load.load_from_text(output_file)
    assert tokenizer.token2id == tokenizer_load.token2id
    # same as above
    assert tokenizer_load.text_to_sequence("猫を飼っています") == [7, 2, 3, 4, 5, 6]
    assert tokenizer_load.sequence_to_text([7, 2, 3, 4, 5, 6]) == "猫を飼っています"

    # equivalent to the original class
    assert tokenizer.token2id == tokenizer_load.token2id
    assert tokenizer.id2token == tokenizer_load.id2token
    assert tokenizer.word_index == tokenizer_load.word_index
    assert tokenizer.set_token == tokenizer_load.set_token


def test_tokenizer_io_pkl():
    text = ["犬を飼っています", "猫を飼っています"]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    tmp_dir = ".pytest_cache/"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    output_file = os.path.join(tmp_dir, "test_tokenizer.pkl")

    # save
    tokenizer.save_as_pkl(output_file)
    # load
    tokenizer_load = Tokenizer()
    tokenizer_load.load_from_pkl(output_file)

    # equivalent to the original class
    assert tokenizer.token2id == tokenizer_load.token2id
    assert tokenizer.id2token == tokenizer_load.id2token
    assert tokenizer.word_index == tokenizer_load.word_index
    assert tokenizer.set_token == tokenizer_load.set_token


if __name__ == "__main__":
    pytest.main([__file__])
