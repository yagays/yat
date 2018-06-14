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


if __name__ == "__main__":
    pytest.main([__file__])
