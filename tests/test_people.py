from __future__ import annotations

import numpy as np
import pytest

from ground_control.people import People

SR = 16000


def test_add_rename_and_things(tmp_path, vad):
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    with pytest.raises(ValueError):
        people.add("ana")
    people.update(ana.id, name="Ana M", things={"todo_list": "todo.ana", "calendar": "calendar.ana"})
    again = People(tmp_path, vad).get(ana.id)
    assert again.name == "Ana M"
    assert again.things == {"todo_list": "todo.ana", "calendar": "calendar.ana"}
    people.update(ana.id, things={"calendar": None})
    assert people.get(ana.id).things == {"todo_list": "todo.ana"}
    with pytest.raises(ValueError):
        people.update(ana.id, things={"car": "x"})


def test_colors_are_distinct(tmp_path, vad):
    people = People(tmp_path, vad)
    colors = [people.add(n).color for n in ("A", "B", "C", "D")]
    assert len(set(colors)) == 4


def test_enroll_needs_speech(tmp_path, vad, embedder):
    people = People(tmp_path, vad)
    p = people.add("Ana")
    with pytest.raises(ValueError):
        people.enroll(p.id, np.zeros(16000 * 5, dtype=np.float32), {"accurate": embedder})


def test_rebuild_matches_enrollment(tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    p = people.add("Ana")
    people.enroll(p.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    _, before = people.voiceprints("accurate")
    people.rebuild("accurate", embedder)
    _, after = people.voiceprints("accurate")
    assert float(before[0] @ after[0]) > 0.999


def test_commands_added_on_their_own_are_cleared_once(tmp_path, speakers, voice, vad, embedder):
    """Up to 2026.11.17 confident commands were folded into voiceprints. An
    install that has them drops them and builds those voiceprints again from
    the enrollments alone."""
    from ground_control.audio import write_wav
    from ground_control.people import window_embeddings

    people = People(tmp_path, vad)
    ana, leo = people.add("Ana"), people.add("Leo")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    people.enroll(leo.id, voice[speakers[1]]["enroll"], {"accurate": embedder})
    clean = dict(zip(*people.voiceprints("accurate")))
    # As the old version left it: a learned command on disk and folded into Ana's voiceprint, here someone else's voice.
    other = voice[speakers[2]]["test"][:4 * SR]
    write_wav(tmp_path / "voices" / ana.id / "learned" / "1.wav", other)
    people._fold("accurate", ana.id, window_embeddings(embedder, other) * 30)
    assert float(dict(zip(*people.voiceprints("accurate")))[ana.id] @ clean[ana.id]) < 0.9

    again = People(tmp_path, vad)
    assert again.clear_learned() == [ana.id]
    assert not (tmp_path / "voices" / ana.id / "learned").exists() and again.needs_rebuild("accurate")
    assert leo.id in again.voiceprints("accurate")[0] and ana.id not in again.voiceprints("accurate")[0]
    again.rebuild("accurate", embedder)
    rebuilt = dict(zip(*again.voiceprints("accurate")))
    assert float(rebuilt[ana.id] @ clean[ana.id]) > 0.999 and float(rebuilt[leo.id] @ clean[leo.id]) > 0.999
    assert again.clear_learned() == []          # nothing to do the next time


def test_forget_removes_voice(tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    p = people.add("Ana")
    people.enroll(p.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    people.forget(p.id)
    assert people.get(p.id) is None
    assert people.voiceprints("accurate")[0] == []
    assert not (tmp_path / "voices" / p.id).exists()
