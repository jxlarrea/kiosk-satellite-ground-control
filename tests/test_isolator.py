"""The Voice Isolator, on its own and in follow mode.

Follow mode holds a turn's audio and sends it at the end of the turn, cleaned.
The real model is checked on a voice over another voice talking at the same
level, the case follow mode alone can't fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from ground_control.follow import Follow, FollowConfig
from ground_control.isolator import WINDOW, Isolator
from ground_control.models import ISOLATOR_FILE, SPEAKER_MODELS
from ground_control.vad import FRAME

from .conftest import MODELS, SR, chunks
from .test_follow import Script, held_out, household, of, silence  # noqa: F401 - household is a fixture

ACCURATE = SPEAKER_MODELS["accurate"]


class Halves:
    """Stands in for the isolator: half the level, and what it was given."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, np.ndarray]] = []

    def clean(self, audio: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        self.calls.append((len(audio), voiceprint.copy()))
        return audio * 0.5


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    proj = (est @ ref) / (ref @ ref) * ref
    return float(10 * np.log10((proj @ proj) / ((est - proj) @ (est - proj)) + 1e-12))


def test_a_turn_goes_out_cleaned_when_it_ends(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("quiet", silence(0.3)).add("user", user[2 * SR:5 * SR]).add("tv", tv[:2 * SR])
         .add("user", user[6 * SR:8 * SR]).add("quiet", silence(1.5)))
    iso = Halves()
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)], isolator=iso)
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    speech = [e for e in events if not e.get("keepalive")]
    kinds = [e["type"] for e in speech]
    # Only a little silence goes out during a turn: each turn's audio comes
    # right before its end, in pieces of half a second at most.
    for i, k in enumerate(kinds):
        if k == "audio":
            assert kinds[i + 1] in ("audio", "turn_end")
    assert max(len(e["audio"]) for e in of(speech, "audio")) <= SR // 2
    assert len(iso.calls) == len(of(events, "turn_end"))
    # The isolator gets the user's own voiceprint, and what goes out is its
    # output cut to what follow mode released.
    assert np.allclose(iso.calls[0][1], f._ref)
    sent = np.concatenate([e["audio"] for e in of(speech, "audio")])
    raw = s.audio
    expected = np.concatenate([raw[a:b] for a, b in f.released]) * 0.5
    assert np.allclose(sent, expected)
    # A turn after the first is held with a little of the speech before it (``judge_lead_s``),
    # here the TV's last half second: the real Isolation takes that out, this stand-in doesn't.
    assert s.released_from("tv", f.released) <= 0.7
    assert "isolated_ms" in of(events, "stats")[0]
    # The saved audio is the speech alone, without the silence.
    assert np.allclose(f.cleaned_audio(), expected)


class Absent:
    """An isolator that takes the user for absent in some turns: near silence
    for those, as the real model answers a voiceprint it doesn't hear."""

    def __init__(self, turns: set[int], level: float = 0.005) -> None:
        self.turns, self.level, self.calls = turns, level, 0

    def clean(self, audio: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        self.calls += 1
        return audio * (self.level if self.calls in self.turns else 0.5)


def test_a_turn_isolation_returns_as_near_silence_goes_on_as_it_was(household, speakers, voice, vad, embedder):
    """The Voice Isolator now and then takes the user for absent in a short
    command and returns it 40 dB quieter. Trimming already took those words
    for the user's, so they reach the assistant as Trimming left them."""
    people, _, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("quiet", silence(0.3)).add("user", user[2 * SR:5 * SR]).add("tv", tv[:2 * SR])
         .add("user", user[6 * SR:8 * SR]).add("quiet", silence(1.5)))
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)], isolator=Absent({1}))
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    steps = [p for p in f.timeline if p["kind"] == "isolated"]
    assert len(steps) >= 2
    # The first turn came back 46 dB quieter: not used, and the session says so.
    assert steps[0]["skipped"] is True and steps[0]["kept_db"] == -46
    assert all("skipped" not in p for p in steps[1:])
    assert of(events, "stats")[0]["isolation_skipped"] == 1 and f.isolation_skipped == 1
    sent = np.concatenate([e["audio"] for e in of(events, "audio") if not e.get("keepalive")])
    raw = np.concatenate([s.audio[a:b] for a, b in f.released])
    first = int(round(of(events, "turn_end")[0]["t"] * SR)) - f.released[0][0]
    turn1 = sum(b - a for a, b in f.released if b <= f.released[0][0] + first + 1)
    assert np.allclose(sent[:turn1], raw[:turn1])              # as Trimming left it
    assert np.allclose(sent[turn1:], raw[turn1:] * 0.5)        # the next turn: Isolation's output
    # A turn that is only a little quieter is Isolation doing its work.
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)],
               isolator=Absent({1}, level=0.15))
    events = [e for c in chunks(s.audio) for e in f.feed(c)] + f.finish()
    assert "isolation_skipped" not in of(events, "stats")[0]
    assert not [p for p in f.timeline if p["kind"] == "isolated" and p.get("skipped")]


class Lost:
    """An isolator on a GPU that doesn't answer for some turns (the Vulkan image)."""

    def __init__(self, turns: set[int]) -> None:
        self.turns, self.calls = turns, 0

    def clean(self, audio: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        from ground_control.vulkan import Failed

        self.calls += 1
        if self.calls in self.turns:
            raise Failed("the GPU's process ended (exit code -6)")
        return audio * 0.5


def test_a_turn_the_gpu_doesnt_answer_for_goes_on_as_it_was(household, speakers, voice, vad, embedder):
    """A GPU driver can reset the GPU under Isolation. The command it was
    working on reaches the assistant as Trimming left it, and the next one is
    cleaned again."""
    people, _, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("quiet", silence(0.3)).add("user", user[2 * SR:5 * SR]).add("tv", tv[:2 * SR])
         .add("user", user[6 * SR:8 * SR]).add("quiet", silence(1.5)))
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)], isolator=Lost({1}))
    events = [e for c in chunks(s.audio) for e in f.feed(c)] + f.finish()
    steps = [p for p in f.timeline if p["kind"] == "isolated"]
    assert len(steps) >= 2
    assert steps[0].get("failed") is True and steps[0]["skipped"] is True and "kept_db" not in steps[0]
    assert all("skipped" not in p for p in steps[1:])
    stats = of(events, "stats")[0]
    assert stats["isolation_failed"] == 1 and "isolation_skipped" not in stats
    sent = np.concatenate([e["audio"] for e in of(events, "audio") if not e.get("keepalive")])
    raw = np.concatenate([s.audio[a:b] for a, b in f.released])
    first = int(round(of(events, "turn_end")[0]["t"] * SR)) - f.released[0][0]
    turn1 = sum(b - a for a, b in f.released if b <= f.released[0][0] + first + 1)
    assert np.allclose(sent[:turn1], raw[:turn1])              # as Trimming left it
    assert np.allclose(sent[turn1:], raw[turn1:] * 0.5)        # the next turn: Isolation's output


def test_a_held_turn_keeps_the_stream_open(household, speakers, voice, vad, embedder):
    """Home Assistant Cloud's speech-to-text gives up on a stream that gets
    nothing for about 6 s, so a held turn sends a little silence as it goes."""
    people, _, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.3)).add("user", user[2 * SR:8 * SR]).add("quiet", silence(1.5))
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)], isolator=Halves())
    out, end = [], None
    for i, c in enumerate(chunks(s.audio)):
        for e in f.feed(c):
            now = (i + 1) * len(c) / SR
            if e["type"] == "audio":
                out.append((now, bool(e.get("keepalive")), e["audio"]))
            if e["type"] == "keepalive":
                out.append((now, True, None))
            if e["type"] == "turn_end" and end is None:
                end = now
    first = next(t for t, keep, _ in out if keep)
    before_end = [t for t, _, _ in out if t < end]
    # Something goes out at least every 0.6 s until the turn is sent: keepalive
    # events before the turn starts, silence while it is held.
    assert first < 1.5
    assert max(b - a for a, b in zip(before_end, before_end[1:])) <= 0.6
    held = [a for _, keep, a in out if keep and a is not None]
    assert held and all(not a.any() and len(a) == int(0.02 * SR) for a in held)


def test_without_the_isolator_audio_goes_out_as_it_comes(household, speakers, voice, vad, embedder):
    people, _, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.3)).add("user", user[2 * SR:6 * SR]).add("quiet", silence(1.5))
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)])
    first_audio = None
    for i, c in enumerate(chunks(s.audio)):
        if any(e["type"] == "audio" for e in f.feed(c)) and first_audio is None:
            first_audio = (i + 1) * 1280 / SR
    assert first_audio is not None and first_audio < 4.0


def test_long_audio_runs_in_overlapping_windows():
    """Through an identity model, the windows and their fades add up to the input."""
    iso = Isolator.__new__(Isolator)
    iso._lock = __import__("threading").Lock()
    iso.calls = 0
    iso._run = lambda audio, voiceprint: audio
    x = np.random.default_rng(0).standard_normal(int(2.6 * WINDOW)).astype(np.float32) * 0.1
    assert np.allclose(iso.clean(x, np.ones(192, np.float32)), x, atol=1e-6)


@pytest.fixture(scope="module")
def isolator():
    if not (MODELS / ISOLATOR_FILE).is_file():
        pytest.skip("no Voice Isolator model")
    return Isolator(MODELS / ISOLATOR_FILE, threads=4)


def test_the_model_carries_its_license(isolator):
    meta = isolator.session.get_modelmeta().custom_metadata_map
    assert meta["license"].startswith("Kiosk Satellite Ground Control Voice Isolator License")
    assert "Copyright" in meta["copyright"]
    assert isolator.version == meta["version"]


def test_the_user_is_kept_and_another_voice_under_them_removed(household, speakers, voice, isolator):
    people, ana, _ = household
    user, other = held_out(speakers[0])[2 * SR:5 * SR], held_out(speakers[2])[2 * SR:5 * SR]
    other = other * (np.sqrt(np.mean(user ** 2)) / np.sqrt(np.mean(other ** 2)))  # the same level
    mix = (user + other).astype(np.float32)
    ids, rows = people.voiceprints("accurate")
    out = isolator.clean(mix, rows[ids.index(ana.id)])
    assert len(out) == len(mix)
    assert si_sdr(out, user) - si_sdr(mix, user) >= 6.0
    # Its level stays the user's.
    assert abs(20 * np.log10(np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(user ** 2)))) < 3.0


def test_a_short_command_is_isolated_with_the_users_profile(household, speakers, voice, vad, embedder):
    """A command too short to identify alone is identified when it ends, from
    the wake word and the command together. Isolation runs after that, so it
    gets the user's profile, not the weaker guess from the wake word."""
    people, ana, _ = household
    user = held_out(speakers[0])
    anchor, command = user[9 * SR:10 * SR], user[int(10.2 * SR):int(11.1 * SR)]
    s = Script().add("quiet", silence(0.3)).add("user", command).add("quiet", silence(1.5))
    iso = Halves()
    f = Follow(FollowConfig(policy="people", people=(ana.id,)), ACCURATE, embedder, vad, people, anchor=anchor,
               isolator=iso)
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    assert next(t for t in f.timeline if t["kind"] == "anchor")["score"] < ACCURATE.threshold
    assert of(events, "speaker")[-1]["id"] == ana.id
    ids, rows = people.voiceprints("accurate")
    assert len(iso.calls) == 1 and np.allclose(iso.calls[0][1], rows[ids.index(ana.id)])
    # Who it is comes before the speech, and the speech before the end of the
    # turn. Only the keepalive silence can go out earlier.
    kinds = [e["type"] for e in events if not e.get("keepalive")]
    assert kinds.index("speaker") < kinds.index("audio") < kinds.index("turn_end")


def test_each_kiosk_chooses_isolation(cfg, isolator):
    """The service's setting allows Isolation, a kiosk's own switch picks it.
    A kiosk that doesn't say (an older Kiosk Satellite) gets it."""
    from ground_control.service import GroundControl

    vm = GroundControl(cfg)
    try:
        vm.update_settings({"isolator": True})
        assert vm.new_follow().isolator is not None
        assert vm.new_follow(isolation=False).isolator is None
        vm.update_settings({"isolator": False})
        assert vm.new_follow(isolation=True).isolator is None
    finally:
        vm.close()


def test_the_stream_stays_open_before_the_first_speech(household, speakers, voice, vad, embedder):
    """A pause after the wake word sends nothing for seconds: Home Assistant
    Cloud would give up after about 6 s (#4). Until speech goes out, keepalive
    events ask the kiosk to send silence. They aren't audio, so the kiosk still
    knows nobody has talked yet."""
    people, _, _ = household
    user = held_out(speakers[0])
    for isolator in (None, Halves()):
        s = Script().add("quiet", silence(6.0)).add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.5))
        f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)], isolator=isolator)
        out, turn_at = [], None
        for i, c in enumerate(chunks(s.audio)):
            for e in f.feed(c):
                now = (i + 1) * len(c) / SR
                if e["type"] in ("audio", "keepalive"):
                    out.append((now, e))
                if e["type"] == "turn_start" and turn_at is None:
                    turn_at = now
        before = [(t, e) for t, e in out if t < turn_at]
        waiting = [t for t, e in before if e["type"] == "keepalive"]
        assert turn_at > 6.0 and waiting and waiting[0] <= 0.6
        # No audio at all before the turn starts: only keepalive events, every 0.5 s.
        assert all(e["type"] == "keepalive" and e["ms"] == 20 for _, e in before)
        assert max(b - a for a, b in zip([0.0] + waiting, waiting)) <= 0.6


def test_keeping_gpu_memory_is_a_setting_and_does_nothing_on_a_cpu(monkeypatch, tmp_path, isolator):
    from ground_control import config

    monkeypatch.setenv("OPTIONS_FILE", str(tmp_path / "none.json"))
    monkeypatch.delenv("KEEP_GPU_MEMORY", raising=False)
    assert not config.load().keep_gpu_memory
    monkeypatch.setenv("KEEP_GPU_MEMORY", "true")
    assert config.load().keep_gpu_memory
    on_cpu = Isolator(MODELS / ISOLATOR_FILE, threads=2, keep_memory=True)
    assert not on_cpu.keeps_memory and not isolator.keeps_memory


def test_a_held_turn_starts_after_the_end_of_the_wake_word(household, speakers, voice, vad, embedder):
    """With Isolation the turn goes out at its end: the silence in front of a
    command cut from the wake word and the fade are put on the cleaned audio."""
    from .test_follow import talking, wake_end

    people, *_ = household
    user = talking(voice[speakers[0]]["test"], vad)
    audio = np.concatenate([user[:4 * SR], silence(1.5)])
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=held_out(speakers[0])[: int(1.2 * SR)],
               isolator=Halves(), wake_word_end=wake_end(0.30))
    events = []
    for c in chunks(audio):
        events += f.feed(c)
    events += f.finish()
    speech = [e for e in of(events, "audio") if not e.get("keepalive")]
    assert max(len(e["audio"]) for e in speech) <= SR // 2
    sent = np.concatenate([e["audio"] for e in speech])
    cut, lead = int(0.30 * SR), int(0.5 * SR)
    assert f.released[0][0] == cut and f.lead_in_s == 0.5
    assert not sent[:lead].any() and abs(sent[lead]) < 1e-6
    expected = np.concatenate([audio[a:b] for a, b in f.released]) * 0.5
    assert np.allclose(sent[lead + 240:], expected[240:])
    assert np.array_equal(f.cleaned_audio(), sent)


def test_the_end_of_the_wake_word_is_looked_for_only_when_it_can_reach_the_assistant(cfg, monkeypatch):
    """Only for a kiosk that says "Talk right after the wake word" is on, with
    the wake word's text and its clip. Never for a follow-up. The setting
    turns it off for every kiosk."""
    from ground_control.service import GroundControl

    vm = GroundControl(cfg)
    asked = []

    class Finds:
        version = "test"

        def find(self, anchor, head, phrase, language):
            asked.append((phrase, language))

    monkeypatch.setattr(vm, "aligner", lambda: Finds() if vm.align else None)
    anchor = np.zeros(2 * SR, dtype=np.float32)
    head = np.zeros(SR, dtype=np.float32)
    wake = {"wake_word": "Alexa", "seamless": True, "language": "fr"}

    def looked(**kw) -> bool:
        f = vm.new_follow(**kw)
        if f._find_end is None:
            return False
        f._find_end(anchor, head)
        return True

    try:
        assert vm.settings()["aligner"] is True
        assert looked(anchor=anchor, wake=wake) and asked == [("Alexa", "fr")]
        assert not looked(anchor=anchor, wake={**wake, "seamless": False})
        assert not looked(anchor=anchor, wake={"seamless": True})
        assert not looked(anchor=anchor, wake=None)
        assert not looked(anchor=None, wake=wake)
        assert not looked(anchor=anchor, wake=wake, speaker="someone")
        vm.update_settings({"aligner": False})
        assert vm.settings()["aligner"] is False
        assert not looked(anchor=anchor, wake=wake)
        # Kept across a restart.
        again = GroundControl(cfg)
        assert again.align is False
        again.close()
    finally:
        vm.close()


def test_the_wake_word_aligner_loads_with_the_first_command_that_needs_it(cfg):
    from ground_control import aligner
    from ground_control.service import GroundControl

    vm = GroundControl(cfg)
    try:
        assert vm._aligner is None
        if aligner.espeak() is None:
            assert vm.settings()["aligner_missing"] == "espeak-ng" and vm.aligner() is None
            return
        assert vm.settings()["aligner_missing"] is None
        loaded = vm.aligner()
        assert loaded is not None and vm.aligner() is loaded
        assert "Wake Word Aligner License" in loaded.session.get_modelmeta().custom_metadata_map["license"]
    finally:
        vm.close()


def test_short_answers_follow_what_the_kiosk_says_and_the_setting(cfg):
    """A reply to something Home Assistant asked can be a short answer. A
    kiosk that doesn't say how the turn started leaves it to whether a wake
    word came. The setting turns short answers off for every kiosk."""
    from ground_control.service import GroundControl

    vm = GroundControl(cfg)
    try:
        assert vm.settings()["short_answers"] is True
        assert vm.new_follow(asked=True).cfg.short_answers is True
        assert vm.new_follow(asked=False).cfg.short_answers is False
        assert vm.new_follow().cfg.short_answers is None
        vm.update_settings({"short_answers": False})
        assert vm.settings()["short_answers"] is False
        assert vm.new_follow(asked=True).cfg.short_answers is False
        assert vm.new_follow().cfg.short_answers is False
        again = GroundControl(cfg)
        assert again.short_answers is False
        again.close()
    finally:
        vm.close()


def test_a_later_turn_is_judged_on_what_isolation_returns(household, speakers, voice, vad, embedder, isolator):
    """With ``judge_later_turns`` a turn after the first is held, cleaned by
    Isolation and judged on the result before the kiosk hears of it. The
    user's follow-up goes out, a stretch of the TV that the checks alone
    might take for the user does not, and nothing is announced for it."""
    people, _, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("quiet", silence(0.3)).add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.0))
         .add("tv", tv[:4 * SR]).add("quiet", silence(1.0)).add("user", user[6 * SR:8 * SR]).add("quiet", silence(1.5)))
    cfg = FollowConfig()
    f = Follow(cfg, ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)], isolator=isolator)
    events = [e for c in chunks(s.audio) for e in f.feed(c)] + f.finish()
    stats = of(events, "stats")[0]
    assert stats["turns"] == 2 and s.released_from("tv", f.released) < 0.3
    assert s.released_from("user", f.released) > 3.5
    # The second turn was judged, and its start came with its audio once it had passed.
    judged = [p for p in f.timeline if p.get("judged")]
    assert judged and not judged[-1].get("left_out")
    kinds = [e["type"] for e in events if e["type"] in ("turn_start", "turn_end") or (e["type"] == "audio" and not e.get("keepalive"))]
    second = len(kinds) - 1 - kinds[::-1].index("turn_start")
    assert kinds[second + 1] == "audio" and kinds[-1] == "turn_end"
    assert [e["turn"] for e in of(events, "turn_start")] == [1, 2]

    # The same session with the TV's stretch forced over the line a held turn needs: it is held,
    # judged and left out, and the kiosk is told of two turns, never of that one.
    f = Follow(cfg, ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)], isolator=isolator)
    tv_a, tv_b = next((a, b) for who, a, b in s.spans if who == "tv")
    label = f._label

    def as_user(frame, final):
        out = label(frame, final)
        if out == "other" and tv_a + SR // 2 <= frame * FRAME < tv_b - SR // 2 and f.turns >= 1:
            f._label_score = 0.45
            return "user"
        return out

    f._label = as_user
    events = [e for c in chunks(s.audio) for e in f.feed(c)] + f.finish()
    stats = of(events, "stats")[0]
    assert stats["turns"] == 2 and stats["left_out"] >= 1
    assert [e["turn"] for e in of(events, "turn_start")] == [1, 2] and len(of(events, "turn_end")) == 2
    assert s.released_from("tv", f.released) < 0.3 and s.released_from("user", f.released) > 3.5
    sent = sum(len(e["audio"]) for e in of(events, "audio") if not e.get("keepalive"))
    assert sent == sum(b - a for a, b in f.released)

    # Turned off, or without Isolation, nothing is held to be judged.
    f = Follow(FollowConfig(judge_later_turns=False), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)],
               isolator=isolator)
    events = [e for c in chunks(s.audio) for e in f.feed(c)] + f.finish()
    assert not [p for p in f.timeline if p.get("judged")]
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)])
    events = [e for c in chunks(s.audio) for e in f.feed(c)] + f.finish()
    assert not [p for p in f.timeline if p.get("judged")] and of(events, "stats")[0]["turns"] == 2


class Keeps:
    """Stands in for the isolator: keeps what is inside ``spans`` of the
    stream and returns the rest as silence."""

    def __init__(self, spans: list[tuple[int, int]]) -> None:
        self.spans, self.calls, self.follow = spans, 0, None

    def clean(self, audio: np.ndarray, voiceprint: np.ndarray) -> np.ndarray:
        self.calls += 1
        at = self.follow.n - len(audio)
        out = np.zeros_like(audio)
        for a, b in self.spans:
            a, b = max(a - at, 0), min(b - at, len(audio))
            if b > a:
                out[a:b] = audio[a:b]
        return out


def test_a_voice_is_turned_away_at_the_end_of_the_first_turn_on_what_isolation_returns(household, speakers, voice, vad,
                                                                                    embedder):
    """A kiosk that answers chosen people, a wake word that came close to one
    of them without naming them and a TV talking through the wait before the
    command. The TV's two seconds used to turn the session away before the
    user spoke. With ``reject_late`` it is decided when the turn ends, on
    what Isolation returns for it."""
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("quiet", silence(0.3)).add("tv", tv[:3 * SR]).add("user", user[2 * SR:int(3.5 * SR)])
         .add("quiet", silence(1.5)))
    tv_a, tv_b = next((a, b) for who, a, b in s.spans if who == "tv")
    user_a, user_b = next((a, b) for who, a, b in s.spans if who == "user")

    def session(isolator, audio=None, **cfg):
        f = Follow(FollowConfig(policy="people", people=(ana.id,), **cfg), ACCURATE, embedder, vad, people,
                   anchor=user[: int(1.2 * SR)], isolator=isolator)
        if isolator is not None:
            isolator.follow = f
        # The wake word names nobody, the TV is taken for the user and nothing the kiosk heard names them.
        assert f._identified and f.speaker["by"] == "wake_word"
        f._identified, f.speaker, f._pending_events = False, None, []
        f._follow_likely(f._rows @ f._anchor_emb, f._anchor_emb)
        named = f._named_by
        f._named_by = lambda scores, best, by="voice": named(scores, best, by) if f._held_look else None
        f._held_look = False
        look = f._isolated_look

        def isolated_look(events, at, last):
            f._held_look = True
            try:
                return look(events, at, last)
            finally:
                f._held_look = False

        f._isolated_look = isolated_look
        label = f._label

        def as_user(frame, final):
            out = label(frame, final)
            return "user" if out == "other" and f.turns <= 1 else out

        f._label = as_user
        events = [e for c in chunks(s.audio if audio is None else audio) for e in f.feed(c)] + f.finish()
        return f, events

    # As it was: turned away two seconds into the TV, before the user's first word.
    f, events = session(Keeps([(user_a, user_b)]), reject_late=False)
    assert of(events, "stats")[0]["rejected"] and of(events, "rejected")[0]["t"] * SR < user_a
    assert f.isolator.calls == 0

    # Now: the turn runs to its end, Isolation keeps the user's words and those name them.
    iso = Keeps([(user_a, user_b)])
    f, events = session(iso)
    stats = of(events, "stats")[0]
    assert not stats["rejected"] and stats["speaker"]["id"] == ana.id and stats["speaker"]["by"] == "voice"
    looks = [p for p in f.timeline if p["kind"] == "identify" and p.get("isolated")]
    # A first look where the session used to be turned away, with only the TV in so far, and one at the end.
    assert len(looks) == 2 and "score" not in looks[0] and looks[1]["score"] >= ACCURATE.threshold
    # No third run of Isolation: what recognized the user is what goes out.
    assert iso.calls == 2
    sent = np.concatenate([e["audio"] for e in of(events, "audio") if not e.get("keepalive")])
    assert float(np.abs(sent).max()) > 0.01
    assert [e["type"] for e in events if e["type"] in ("speaker", "turn_start", "turn_end", "rejected")][-1] == "turn_end"

    # Isolation hears none of that person in the turn: turned away at its end.
    iso = Keeps([])
    f, events = session(iso)
    assert of(events, "stats")[0]["rejected"] and iso.calls == 2
    assert of(events, "rejected")[0]["t"] * SR >= user_b - SR // 2
    assert not [e for e in of(events, "audio") if not e.get("keepalive")]

    # A voice that talks on and on is turned away at ``reject_late_s`` of speech.
    long_tv = np.concatenate([silence(0.3), np.tile(tv[:4 * SR], 3)])
    iso = Keeps([])
    f, events = session(iso, audio=long_tv)
    assert of(events, "stats")[0]["rejected"] and iso.calls == 2
    assert 6.0 <= of(events, "rejected")[0]["t"] <= 8.0

    # A wake word that isn't close to anyone the kiosk answers gets no more time than before.
    f = Follow(FollowConfig(policy="people", people=(ana.id,)), ACCURATE, embedder, vad, people,
               anchor=tv[4 * SR:int(5.2 * SR)], isolator=Keeps([]))
    f.isolator.follow = f
    assert not f._wake_close
    events = [e for c in chunks(long_tv) for e in f.feed(c)] + f.finish()
    assert f.isolator.calls == 0
    assert of(events, "stats")[0]["rejected"] and not [p for p in f.timeline if p.get("isolated")]
