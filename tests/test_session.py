from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ground_control.models import SPEAKER_MODELS
from ground_control.people import People
from ground_control.session import Session, SessionConfig

from .conftest import SR, chunks

ACCURATE = SPEAKER_MODELS["accurate"]


@pytest.fixture(scope="module")
def household(tmp_path_factory, speakers, voice, vad, embedder):
    """Ana and Leo enrolled. The third speaker is a stranger."""
    people = People(tmp_path_factory.mktemp("people"), vad)
    ana, leo = people.add("Ana"), people.add("Leo")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    people.enroll(leo.id, voice[speakers[1]]["enroll"], {"accurate": embedder})
    return people, ana, leo


def run(folks, embedder, vad, audio, **kw):
    s = Session(SessionConfig(**kw), ACCURATE, embedder, vad, folks)
    events = []
    for c in chunks(audio):
        events += s.feed(c)
    return events + s.finish(), s


def first(events, kind):
    return next((e for e in events if e["type"] == kind), None)


def test_identifies_the_speaker_while_talking(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    audio = np.concatenate([voice[speakers[0]]["test"][:5 * SR], np.zeros(2 * SR, dtype=np.float32)])
    events, _ = run(people, embedder, vad, audio)
    speaker = first(events, "speaker")
    assert speaker["id"] == ana.id and speaker["by"] == "voice"
    assert speaker["t"] < 2.0
    end = first(events, "end_of_speech")
    assert end["reason"] == "silence"
    assert first(events, "other_voice") is None


def test_a_blip_does_not_end_the_turn(household, speakers, voice, vad, embedder):
    """The wake chime's echo or a cough, then quiet, then the command."""
    people, ana, _ = household
    speech = voice[speakers[0]]["test"]
    start = next(i for i in range(0, len(speech), 160) if abs(speech[i:i + 160]).max() > 0.05)
    blip = speech[start:start + int(0.4 * SR)]
    quiet = np.zeros(int(1.2 * SR), dtype=np.float32)
    audio = np.concatenate([blip, quiet, speech[:4 * SR], np.zeros(2 * SR, dtype=np.float32)])
    events, _ = run(people, embedder, vad, audio)
    end = first(events, "end_of_speech")
    assert end["reason"] == "silence"
    assert end["t"] > 1.6 + 1.0
    assert first(events, "speaker")["id"] == ana.id


@pytest.mark.parametrize("other", [1, 2], ids=["enrolled", "stranger"])
def test_other_voice_cuts_at_the_boundary(household, speakers, voice, vad, embedder, other):
    people, ana, _ = household
    boundary = 4.0
    audio = np.concatenate([voice[speakers[0]]["test"][:int(boundary * SR)], voice[speakers[other]]["test"][:4 * SR]])
    events, _ = run(people, embedder, vad, audio)
    assert first(events, "speaker")["id"] == ana.id
    ov = first(events, "other_voice")
    assert ov is not None
    assert abs(ov["cut"] - boundary) <= 0.5
    assert ov["t"] - boundary <= 1.6
    assert first(events, "end_of_speech")["reason"] == "other_voice"


def test_gated_audio_drops_the_other_voice(household, speakers, voice, vad, embedder):
    people, _, _ = household
    boundary = 4.0
    audio = np.concatenate([voice[speakers[0]]["test"][:int(boundary * SR)], voice[speakers[2]]["test"][:4 * SR]])
    events, _ = run(people, embedder, vad, audio, mode="gated")
    out = np.concatenate([e["audio"] for e in events if e["type"] == "audio"])
    cut = first(events, "other_voice")["cut"]
    assert len(out) <= int(cut * SR) + 1
    assert np.array_equal(out, audio[:len(out)])


def test_enrolled_only_rejects_a_stranger(household, speakers, voice, vad, embedder):
    people, _, _ = household
    events, _ = run(people, embedder, vad, voice[speakers[2]]["test"][:5 * SR], policy="enrolled")
    assert first(events, "rejected") is not None
    assert first(events, "speaker")["name"] is None


def test_only_the_chosen_people_are_answered(household, speakers, voice, vad, embedder):
    people, ana, leo = household
    events, _ = run(people, embedder, vad, voice[speakers[1]]["test"][:5 * SR], policy="people", people=(ana.id,))
    assert first(events, "speaker")["name"] == "Leo"
    assert first(events, "rejected") is not None
    events, _ = run(people, embedder, vad, voice[speakers[0]]["test"][:5 * SR], policy="people", people=(ana.id,))
    assert first(events, "speaker")["name"] == "Ana"
    assert first(events, "rejected") is None
    events, _ = run(people, embedder, vad, voice[speakers[2]]["test"][:5 * SR], policy="people", people=(ana.id,))
    assert first(events, "rejected") is not None


def test_anyone_lets_a_stranger_through(household, speakers, voice, vad, embedder):
    people, _, _ = household
    events, _ = run(people, embedder, vad, voice[speakers[2]]["test"][:5 * SR])
    assert first(events, "rejected") is None


def test_each_turn_checks_the_speaker_again(household, speakers, voice, vad, embedder):
    people, ana, leo = household
    s = Session(SessionConfig(), ACCURATE, embedder, vad, people)
    gap = np.zeros(SR, dtype=np.float32)
    turn1, turn2 = [], []
    for c in chunks(np.concatenate([voice[speakers[0]]["test"][:4 * SR], gap])):
        turn1 += s.feed(c)
    s.new_turn()
    for c in chunks(np.concatenate([voice[speakers[1]]["test"][:4 * SR], gap])):
        turn2 += s.feed(c)
    assert first(turn1, "speaker")["id"] == ana.id
    assert first(turn2, "speaker")["id"] == leo.id


def test_only_person_without_a_voice(tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    sam = people.add("Sam")
    events, _ = run(people, embedder, vad, voice[speakers[2]]["test"][:3 * SR])
    speaker = first(events, "speaker")
    assert speaker["id"] == sam.id and speaker["by"] == "only_person"


def test_a_confident_command_is_not_added_to_the_voice(tmp_path, speakers, voice, vad, embedder):
    """A voiceprint is only made from audio someone chose to add. A command,
    however well recognized, carries whatever played in the room with it."""
    people = People(tmp_path, vad)
    ana, leo = people.add("Ana"), people.add("Leo")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    people.enroll(leo.id, voice[speakers[1]]["enroll"], {"accurate": embedder})
    before = people.voiceprints("accurate")[1].copy()
    events, _ = run(people, embedder, vad, np.concatenate([voice[speakers[0]]["test"][:5 * SR],
                                                           np.zeros(SR, dtype=np.float32)]))
    assert first(events, "speaker")["id"] == ana.id and "learned" not in first(events, "stats")
    assert np.array_equal(people.voiceprints("accurate")[1], before)
    assert not (tmp_path / "voices" / ana.id / "learned").exists()
