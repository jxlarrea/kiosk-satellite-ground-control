"""Shared fixtures: the real models and LibriSpeech speakers.

Tests that need audio skip when the corpus isn't on this machine. Set
``VM_TEST_SPEECH`` to a folder of ``ls_<speaker>_<chapter>_<utt>.wav`` files
to use another copy.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pytest

from ground_control.audio import read_wav
from ground_control.config import Config
from ground_control.engine import Embedder
from ground_control.models import SPEAKER_MODELS
from ground_control.vad import VadModel

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SPEECH = Path(os.environ.get("VM_TEST_SPEECH", "/mnt/training_data/data/corpora/librispeech/dev-clean"))
SR = 16000


def clips(speaker: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(SPEECH / f"ls_{speaker}_*.wav")))]


@pytest.fixture(scope="session")
def speakers() -> list[str]:
    if not SPEECH.is_dir():
        pytest.skip("no test speech")
    return sorted({Path(p).name.split("_")[1] for p in glob.glob(str(SPEECH / "ls_*_*.wav"))})[:3]


@pytest.fixture(scope="session")
def voice(speakers):
    """Enrollment audio (8 clips) and a held-out test clip per speaker."""
    out = {}
    for s in speakers:
        c = clips(s)
        out[s] = {"enroll": np.concatenate([read_wav(p) for p in c[:8]]), "test": read_wav(c[-1])}
    return out


@pytest.fixture(scope="session")
def vad() -> VadModel:
    return VadModel(MODELS / "silero_vad.onnx")


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    return Embedder(MODELS / SPEAKER_MODELS["accurate"].file, threads=2)


@pytest.fixture
def cfg(tmp_path_factory) -> Config:
    return Config(data_dir=tmp_path_factory.mktemp("data"), models_dir=MODELS, model="accurate", threads=2,
                  cuda=False, ha_url="http://ha.invalid", port=0, auth="homeassistant")


def chunks(audio: np.ndarray, size: int = 1280):
    for i in range(0, len(audio), size):
        yield audio[i:i + size]
