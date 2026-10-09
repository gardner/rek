"""Match timestamped words to diarization turns and format a readable transcript."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isclose


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Word:
    start: float | None
    end: float | None
    text: str


@dataclass(frozen=True)
class Utterance:
    start: float | None
    end: float | None
    speaker: str
    text: str


def _speaker_for(word: Word, turns: Sequence[SpeakerTurn]) -> str:
    if word.start is None or word.end is None or word.end < word.start:
        return "Unknown"
    scores: dict[str, float] = defaultdict(float)
    for turn in turns:
        overlap = max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
        if word.start == word.end and turn.start <= word.start < turn.end:
            overlap = 1.0
        if overlap > 0:
            scores[turn.speaker] += overlap
    if not scores:
        return "Unknown"
    best = max(scores.values())
    winners = [speaker for speaker, score in scores.items() if isclose(score, best)]
    return winners[0] if len(winners) == 1 else "Unknown"


def align_words(words: Sequence[Word], turns: Sequence[SpeakerTurn]) -> list[Utterance]:
    """Assign by total temporal overlap; retain ambiguous words as Unknown."""
    utterances: list[Utterance] = []
    for word in words:
        if not word.text.strip():
            continue
        speaker = _speaker_for(word, turns)
        previous = utterances[-1] if utterances else None
        if (
            previous is not None
            and previous.speaker == speaker
            and previous.end is not None
            and word.start is not None
            and word.end is not None
            and 0 <= word.start - previous.end <= 1.0
        ):
            # Whisper retains leading spaces and punctuation in its word chunks.
            utterances[-1] = replace(previous, end=word.end, text=previous.text + word.text)
        else:
            utterances.append(Utterance(word.start, word.end, speaker, word.text.strip()))
    return utterances


def _timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "??:??:??.???"
    total_ms = round(seconds * 1000)
    minutes, ms = divmod(total_ms, 60_000)
    hours, minutes = divmod(minutes, 60)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"


def render_transcript(utterances: Sequence[Utterance]) -> str:
    if not utterances:
        return "No speech detected.\n"
    return "".join(
        f"[{_timestamp(line.start)} - {_timestamp(line.end)}] {line.speaker}: {line.text}\n"
        for line in utterances
    )
