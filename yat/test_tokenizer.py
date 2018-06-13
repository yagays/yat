import pytest
from yat import Tokenizer


def test_tokenizer():
    text = ["犬を飼っています", "猫を飼っています"]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    assert tokenizer.num_id == 7
    assert tokenizer.texts_to_sequences(text) == [[0, 1, 2, 3, 4, 5], [6, 1, 2, 3, 4, 5]]
    assert tokenizer.sequences_to_texts([[0, 1, 2, 3, 4, 5], [6, 1, 2, 3, 4, 5]]) == text

    # unknown word: "狼"
    assert tokenizer.text_to_sequence("狼を飼っています") == [-1, 1, 2, 3, 4, 5]
    assert tokenizer.sequence_to_text([-1, 1, 2, 3, 4, 5]) == "UNKを飼っています"

    # fit additional word "狼"
    tokenizer.fit_on_text("狼を飼っています")
    assert tokenizer.num_id == 8
    assert tokenizer.text_to_sequence("狼を飼っています") == [7, 1, 2, 3, 4, 5]

    tokenizer.fit_on_texts(["狐を飼っています", "狸を飼っています"])
    assert tokenizer.num_id == 10
    assert tokenizer.texts_to_sequences(["狐を飼っています", "狸を飼っています"]) == [[8, 1, 2, 3, 4, 5], [9, 1, 2, 3, 4, 5]]

if __name__ == "__main__":
    pytest.main([__file__])
