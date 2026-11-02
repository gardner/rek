# Cache policy adapted from Hugging Face Transformers.
# Copyright 2026 The HuggingFace Inc. team. Licensed under Apache-2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""NumPy arrival-order speaker cache for the pinned Nemotron ONNX export.

Policy follows Hugging Face's Nemotron3DiarizationSpeakerCache:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py
"""

import math

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float32]
CACHE_LENGTH = 264
FIFO_LENGTH = 40
UPDATE_PERIOD = 300
NUM_SPEAKERS = 8
SUBSAMPLING = 8


class SpeakerCache:
    """Keep representative speaker frames plus a FIFO across offline chunks."""

    def __init__(self) -> None:
        self.speakers: FloatArray = np.empty((0, 512), dtype=np.float32)
        self.probabilities: FloatArray = np.empty((0, NUM_SPEAKERS), dtype=np.float32)
        self.fifo: FloatArray = np.empty((0, 512), dtype=np.float32)
        self.compressed = False

    @property
    def embeds(self) -> FloatArray:
        return np.concatenate((self.speakers, self.fifo))

    def update(self, chunk: FloatArray, logits: FloatArray, silence: FloatArray) -> None:
        """Append core frames only; look-ahead frames must not enter the cache."""
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        pooled = (
            probabilities.reshape(-1, SUBSAMPLING, NUM_SPEAKERS).mean(axis=1).astype(np.float32)
        )
        cached = len(self.speakers)
        fifo = np.concatenate((self.fifo, chunk))
        if len(fifo) <= FIFO_LENGTH:
            self.fifo = fifo
            return
        popped = min(len(fifo), max(UPDATE_PERIOD, len(fifo) - FIFO_LENGTH))
        stored = self.probabilities if self.compressed else pooled[:cached]
        self.speakers = np.concatenate((self.speakers, fifo[:popped]))
        self.probabilities = np.concatenate((stored, pooled[cached : cached + popped]))
        self.fifo = fifo[popped:]
        if len(self.speakers) > CACHE_LENGTH:
            self._compress(silence)
            self.compressed = True

    def _compress(self, silence: FloatArray) -> None:
        scores = _frame_scores(self.probabilities)
        scores[CACHE_LENGTH:] += 0.05
        # Each speaker has a 32-frame budget plus one reserved silence frame.
        for count, boost in ((24, -2 * math.log(0.5)), (48, -math.log(0.5))):
            indices = np.argsort(scores, axis=0, kind="stable")[-count:]
            np.add.at(scores, (indices, np.arange(NUM_SPEAKERS)), boost)
        scores = np.concatenate((scores, np.full((1, NUM_SPEAKERS), np.inf)))
        frames = len(scores)
        flat = scores.T.reshape(-1)
        selected = np.argsort(flat, kind="stable")[-CACHE_LENGTH:]
        selected = np.where(np.isneginf(flat[selected]), frames * NUM_SPEAKERS, selected)
        selected.sort()
        indices = np.where(selected == frames * NUM_SPEAKERS, frames - 1, selected % frames)
        self.speakers = np.concatenate((self.speakers, silence.reshape(1, -1)))[indices]
        self.probabilities = np.concatenate(
            (self.probabilities, np.zeros((1, NUM_SPEAKERS), dtype=np.float32))
        )[indices]


def _frame_scores(probabilities: FloatArray) -> FloatArray:
    log_probabilities = np.log(np.maximum(probabilities, 0.25))
    log_complements = np.log(np.maximum(1 - probabilities, 0.25))
    scores = log_probabilities - log_complements + log_complements.sum(axis=1, keepdims=True)
    scores -= math.log(0.5)
    scores[probabilities <= 0.5] = -np.inf
    positive = scores > 0
    enough_positive = positive.sum(axis=0) >= 16
    scores[(~positive) & enough_positive[None, :]] = -np.inf
    return scores
