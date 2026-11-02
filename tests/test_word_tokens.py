"""Reconstruct Parakeet words before assigning each complete word to a speaker."""

from rek.transcript import SpeakerTurn, Word, align_words, merge_word_tokens


def test_subwords_and_punctuation_share_one_word_timestamp() -> None:
    pieces = [Word(0.0, 0.1, " Hel"), Word(0.1, 0.3, "lo"), Word(0.3, 0.4, ",")]
    assert merge_word_tokens(pieces) == [Word(0.0, 0.4, " Hello,")]


def test_standalone_whitespace_starts_the_next_word_at_its_first_spoken_token() -> None:
    pieces = [
        Word(0.0, 0.3, "Hi"),
        Word(0.3, 0.4, " "),
        Word(0.5, 0.6, "there"),
        Word(0.6, 0.7, "."),
    ]
    assert merge_word_tokens(pieces) == [Word(0.0, 0.3, " Hi"), Word(0.5, 0.7, " there.")]
    assert merge_word_tokens([]) == []


def test_empty_pieces_do_not_split_a_word() -> None:
    pieces = [Word(0.0, 0.2, " test"), Word(0.2, 0.2, ""), Word(0.2, 0.4, "ing")]
    assert merge_word_tokens(pieces) == [Word(0.0, 0.4, " testing")]


def test_speaker_change_cannot_split_a_word_or_lose_spacing() -> None:
    pieces = [
        Word(0.0, 0.1, " Hel"),
        Word(0.1, 0.3, "lo,"),
        Word(0.3, 0.5, " world"),
        Word(0.5, 0.6, "!"),
    ]
    turns = [SpeakerTurn(0.0, 0.1, "Speaker 1"), SpeakerTurn(0.1, 0.6, "Speaker 2")]
    result = align_words(merge_word_tokens(pieces), turns)
    assert len(result) == 1
    assert result[0].speaker == "Speaker 2"
    assert result[0].text == "Hello, world!"
