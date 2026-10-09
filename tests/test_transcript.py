"""Speaker assignment must preserve words without inventing speaker identities."""

from tscribe.transcript import SpeakerTurn, Utterance, Word, align_words, render_transcript


def test_groups_words_and_splits_at_speaker_changes() -> None:
    words = [Word(0.1, 0.5, " Hello"), Word(0.5, 0.9, " there."), Word(1.1, 1.5, " Hi!")]
    turns = [SpeakerTurn(0.0, 1.0, "Speaker 1"), SpeakerTurn(1.0, 2.0, "Speaker 2")]

    assert align_words(words, turns) == [
        Utterance(0.1, 0.9, "Speaker 1", "Hello there."),
        Utterance(1.1, 1.5, "Speaker 2", "Hi!"),
    ]


def test_adjacent_words_ignore_timestamp_rounding_errors() -> None:
    words = [Word(2.0, 2.4800000000000004, " Hello"), Word(2.48, 3.0, " there.")]
    turns = [SpeakerTurn(0.0, 4.0, "Speaker 1")]
    assert align_words(words, turns) == [Utterance(2.0, 3.0, "Speaker 1", "Hello there.")]


def test_actually_overlapping_words_remain_separate() -> None:
    words = [Word(0.0, 0.5, " One"), Word(0.4, 0.8, " two")]
    turns = [SpeakerTurn(0.0, 1.0, "Speaker 1")]
    assert len(align_words(words, turns)) == 2


def test_assigns_by_total_overlap_per_speaker() -> None:
    turns = [
        SpeakerTurn(0.0, 0.3, "Speaker 1"),
        SpeakerTurn(0.7, 1.0, "Speaker 1"),
        SpeakerTurn(0.3, 0.7, "Speaker 2"),
    ]
    assert align_words([Word(0.0, 1.0, " hello")], turns)[0].speaker == "Speaker 1"


def test_unmatched_or_missing_timestamps_remain_unknown() -> None:
    words = [Word(5.0, 5.2, " unmatched"), Word(None, None, " missing")]
    turns = [SpeakerTurn(0.0, 1.0, "Speaker 1")]
    result = align_words(words, turns)

    assert [line.speaker for line in result] == ["Unknown", "Unknown"]
    assert result[1].start is None
    assert result[1].text == "missing"


def test_equal_overlap_is_ambiguous_and_word_is_not_duplicated() -> None:
    turns = [SpeakerTurn(0.0, 1.0, "Speaker 2"), SpeakerTurn(0.0, 1.0, "Speaker 1")]
    assert align_words([Word(0.1, 0.9, " hello")], turns) == [
        Utterance(0.1, 0.9, "Unknown", "hello")
    ]


def test_zero_duration_word_uses_containing_turn() -> None:
    turns = [SpeakerTurn(0.0, 1.0, "Speaker 1"), SpeakerTurn(1.0, 2.0, "Speaker 2")]
    assert align_words([Word(1.0, 1.0, " Yes")], turns)[0].speaker == "Speaker 2"


def test_long_pauses_split_utterances_and_empty_words_are_skipped() -> None:
    words = [Word(0.1, 0.5, " Hi"), Word(0.5, 0.6, " "), Word(3.0, 3.5, " again")]
    turns = [SpeakerTurn(0.0, 4.0, "Speaker 1")]
    assert len(align_words(words, turns)) == 2
    assert align_words([], turns) == []


def test_rendering_keeps_timestamps_and_unknown_times() -> None:
    assert render_transcript([Utterance(61.25, 62.0, "Speaker 1", "Hello.")]) == (
        "[00:01:01.250 - 00:01:02.000] Speaker 1: Hello.\n"
    )
    assert "Unknown: Hi" in render_transcript([Utterance(None, None, "Unknown", "Hi")])
    assert render_transcript([]) == "No speech detected.\n"
