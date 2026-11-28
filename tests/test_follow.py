"""Follow mode on recorded speech: only the user's segments go on.

Ana and Leo are enrolled. The third speaker plays the TV, or a guest. The
anchor stands in for the wake word: a short clip of the user that is not
part of their enrollment.
"""

from __future__ import annotations

import numpy as np
import pytest

from ground_control.audio import read_wav
from ground_control.follow import Follow, FollowConfig
from ground_control.models import SPEAKER_MODELS
from ground_control.people import People
from ground_control.vad import FRAME

from .conftest import SR, chunks, clips

ACCURATE = SPEAKER_MODELS["accurate"]


@pytest.fixture(scope="module")
def household(tmp_path_factory, speakers, voice, vad, embedder):
    people = People(tmp_path_factory.mktemp("people"), vad)
    ana, leo = people.add("Ana"), people.add("Leo")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    people.enroll(leo.id, voice[speakers[1]]["enroll"], {"accurate": embedder})
    return people, ana, leo


def held_out(speaker: str) -> np.ndarray:
    """Speech from neither the enrollment nor the test clip."""
    c = clips(speaker)
    return np.concatenate([read_wav(p) for p in c[8:-1]]) if len(c) > 9 else read_wav(c[-2])


def anchor_of(audio: np.ndarray) -> np.ndarray:
    return audio[: int(1.2 * SR)]


def silence(s: float) -> np.ndarray:
    return np.zeros(int(s * SR), dtype=np.float32)


class Script:
    """Audio built from labeled parts, with where each part landed."""

    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []
        self.spans: list[tuple[str, int, int]] = []
        self.n = 0

    def add(self, who: str, audio: np.ndarray) -> Script:
        self.parts.append(audio.astype(np.float32))
        self.spans.append((who, self.n, self.n + len(audio)))
        self.n += len(audio)
        return self

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate(self.parts)

    def released_from(self, who: str, released: list[tuple[int, int]]) -> float:
        """Seconds of ``who`` that were released."""
        total = 0
        for name, a, b in self.spans:
            if name != who:
                continue
            for ra, rb in released:
                total += max(0, min(b, rb) - max(a, ra))
        return total / SR

    def seconds(self, who: str) -> float:
        return sum(b - a for name, a, b in self.spans if name == who) / SR

    def speech_share(self, who: str, released: list[tuple[int, int]], vad) -> float:
        """The share of ``who``'s speech (VAD frames, not their pauses) that was
        released: pauses inside a turn go out shortened, the words go out whole."""
        probs = vad.stream().feed(self.audio)
        frames = [(i * FRAME, (i + 1) * FRAME) for i, p in enumerate(probs) if p >= 0.5]
        mine = [(a, b) for a, b in frames if any(n == who and s <= a and b <= e for n, s, e in self.spans)]
        if not mine:
            return 0.0
        return sum(1 for a, b in mine if any(ra <= a and b <= rb for ra, rb in released)) / len(mine)


def run(folks, embedder, vad, script: Script, anchor=None, **kw):
    f = Follow(FollowConfig(**kw), ACCURATE, embedder, vad, folks, anchor=anchor)
    events = []
    for c in chunks(script.audio):
        events += f.feed(c)
    events += f.finish()
    return events, f


def of(events, kind):
    return [e for e in events if e["type"] == kind]


def test_tv_before_between_and_after_the_user_is_left_out(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("quiet", silence(0.3)).add("tv", tv[:2 * SR]).add("user", user[2 * SR:5 * SR])
         .add("tv", tv[2 * SR:4 * SR]).add("user", user[6 * SR:8 * SR]).add("tv", tv[4 * SR:6 * SR])
         .add("quiet", silence(1.5)))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user))
    assert of(events, "speaker")[0]["id"] == ana.id
    assert of(events, "speaker")[0]["by"] == "wake_word"
    # Each TV stretch is longer than a pause: the user's two stretches are two turns.
    assert len(of(events, "turn_start")) == 2 and len(of(events, "turn_end")) == 2
    assert s.speech_share("user", f.released, vad) >= 0.8
    assert s.released_from("tv", f.released) <= 0.5
    released = sum(len(e["audio"]) for e in of(events, "audio") if not e.get("keepalive")) / SR
    assert released == pytest.approx(sum(b - a for a, b in f.released) / SR, abs=0.01)


def test_tv_right_after_the_wake_word_is_not_the_user(household, speakers, voice, vad, embedder):
    """Without the anchor, the TV talking first would be taken for the user."""
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("tv", tv[:2500 * SR // 1000]).add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user))
    assert len(of(events, "turn_start")) == 1
    assert s.released_from("tv", f.released) <= 0.3
    assert s.speech_share("user", f.released, vad) >= 0.8
    end = of(events, "turn_end")[0]
    user_end = next(b for who, a, b in s.spans if who == "user") / SR
    assert end["t"] == pytest.approx(user_end, abs=0.5)


def test_the_turn_ends_when_the_user_stops_even_with_the_tv_on(household, speakers, voice, vad, embedder):
    people, _, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("user", user[2 * SR:5 * SR]).add("tv", tv[: 6 * SR])
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=anchor_of(user))
    ended_at = None
    fed = 0
    for c in chunks(s.audio):
        fed += len(c)
        if any(e["type"] == "turn_end" for e in f.feed(c)) and ended_at is None:
            ended_at = fed / SR
    assert ended_at is not None, "the TV kept the turn open"
    # The user stopped at 3 s: the turn ends well before the TV does.
    assert ended_at < 3.0 + 2.0
    assert s.released_from("tv", f.released) <= 0.5


def test_a_guest_is_followed_by_the_wake_word(household, speakers, voice, vad, embedder):
    """Nobody enrolled said it: the anchor's own voice is the user, Ana on the TV is not.

    Leo's voice is too close to this guest's (0.42) to tell apart from a guest's
    wake word: an enrolled voiceprint is what separates voices that close."""
    people, _, _ = household
    guest, tv = held_out(speakers[2]), voice[speakers[0]]["test"]
    s = (Script().add("user", guest[2 * SR:5 * SR]).add("tv", tv[:3 * SR]).add("user", guest[6 * SR:8 * SR])
         .add("quiet", silence(1.5)))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(guest))
    assert of(events, "speaker")[-1]["name"] is None
    assert s.released_from("tv", f.released) <= 0.5
    assert s.speech_share("user", f.released, vad) >= 0.7
    assert not of(events, "rejected")


def test_enrolled_only_rejects_a_guest_and_releases_nothing_after(household, speakers, voice, vad, embedder):
    people, _, _ = household
    guest = held_out(speakers[2])
    s = Script().add("user", guest[2 * SR:6 * SR]).add("quiet", silence(1.5)).add("user", guest[6 * SR:8 * SR])
    events, f = run(people, embedder, vad, s, anchor=anchor_of(guest), policy="enrolled")
    rejected = of(events, "rejected")
    assert rejected
    after = [e for e in events if e["type"] == "audio" and e["t"] > rejected[0]["t"] + 0.1]
    assert not after


def test_without_an_anchor_the_first_voice_is_the_user(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("user", user[2 * SR:5 * SR]).add("tv", tv[:3 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s)
    assert of(events, "speaker")[0]["id"] == ana.id
    assert s.released_from("tv", f.released) <= 0.5


def test_a_pause_is_shortened_but_kept(household, speakers, voice, vad, embedder):
    people, _, _ = household
    user = held_out(speakers[0])
    s = Script().add("user", user[2 * SR:4 * SR]).add("quiet", silence(1.0)).add("user", user[4 * SR:6 * SR])
    s.add("quiet", silence(2.0))
    # A long end of turn, so the 1 s pause stays inside one turn.
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user), eos_silence_s=1.6)
    assert len(of(events, "turn_start")) == 1
    released = sum(b - a for a, b in f.released) / SR
    assert s.speech_share("user", f.released, vad) >= 0.9
    # The 1 s pause went as 0.15 s, under the 0.25 s of silence that ends a command
    # in Home Assistant's aggressive end of speech.
    assert released < s.seconds("user") + 0.25


def test_without_a_wake_word_the_household_is_the_user(tmp_path, speakers, voice, vad, embedder):
    """A manual start has no anchor: the TV talking first is not taken for the user.

    Only Ana is enrolled here: this TV voice is close to Leo's (0.42) and
    would pass as him."""
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("tv", tv[: int(1.5 * SR)]).add("user", user[2 * SR:5 * SR]).add("tv", tv[2 * SR:int(4.5 * SR)])
         .add("quiet", silence(1.0)))
    events, f = run(people, embedder, vad, s)
    assert of(events, "speaker")[0]["id"] == ana.id
    assert s.released_from("tv", f.released) <= 0.5
    assert s.speech_share("user", f.released, vad) >= 0.8


def test_without_a_wake_word_a_voice_nobody_enrolled_is_answered(tmp_path, speakers, voice, vad, embedder):
    """Home Assistant asked a question, so there is no wake word, and the person
    answering isn't enrolled: a device that answers anyone follows them."""
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    guest = held_out(speakers[2])
    s = Script().add("quiet", silence(0.5)).add("user", guest[2 * SR:6 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s)
    assert len(of(events, "turn_start")) == 1
    assert s.speech_share("user", f.released, vad) >= 0.8
    assert of(events, "speaker")[-1]["name"] is None


def test_without_a_wake_word_a_short_answer_goes_through(tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    guest = held_out(speakers[2])
    word = guest[2 * SR:int(2.7 * SR)]
    s = Script().add("quiet", silence(0.5)).add("user", word).add("quiet", silence(1.5))
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people)
    ended = False
    for c in chunks(s.audio):
        ended = ended or any(e["type"] == "turn_end" for e in f.feed(c))
    assert ended, "the answer waited for the end of the session"
    assert s.speech_share("user", f.released, vad) >= 0.5


def test_enrolled_only_without_a_wake_word_leaves_a_voice_nobody_enrolled_out(tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    guest = held_out(speakers[2])
    s = Script().add("user", guest[2 * SR:6 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, policy="enrolled")
    assert s.released_from("user", f.released) <= 0.3


def test_only_the_chosen_people_are_followed(household, speakers, voice, vad, embedder):
    """Ana says the wake word on a device that answers only Leo: nothing goes on."""
    people, ana, leo = household
    user = held_out(speakers[0])
    s = Script().add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user), policy="people", people=(leo.id,))
    assert of(events, "speaker")[0]["name"] == "Ana"
    assert of(events, "rejected")
    assert not f.released
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user), policy="people", people=(ana.id,))
    assert not of(events, "rejected")
    assert s.speech_share("user", f.released, vad) >= 0.8


def test_without_a_wake_word_only_the_chosen_people_are_the_user(household, speakers, voice, vad, embedder):
    people, ana, leo = household
    user = held_out(speakers[0])
    s = Script().add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, policy="people", people=(leo.id,))
    assert s.released_from("user", f.released) <= 0.3


def test_a_short_command_after_a_short_wake_word_is_still_recognized(household, speakers, voice, vad, embedder):
    """Neither the wake word (under the threshold on its own) nor a command
    under a second is enough alone. Together they are, so a device that
    answers only Ana doesn't turn her away."""
    people, ana, _ = household
    user = held_out(speakers[0])
    anchor, command = user[9 * SR:10 * SR], user[int(10.2 * SR):int(11.1 * SR)]
    s = Script().add("quiet", silence(0.3)).add("user", command).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=anchor, policy="people", people=(ana.id,))
    assert next(t for t in f.timeline if t["kind"] == "anchor")["score"] < ACCURATE.threshold
    assert of(events, "speaker")[-1]["id"] == ana.id
    assert not of(events, "rejected")
    # The score that recognized her is on the timeline, with the wake word in it.
    found = [t for t in f.timeline if t["kind"] == "identify"]
    assert found[-1]["score"] >= ACCURATE.threshold and found[-1]["wake_word"] is True


def test_a_short_command_from_a_guest_is_still_turned_away(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    guest = held_out(speakers[2])
    anchor, command = guest[9 * SR:10 * SR], guest[int(10.2 * SR):int(11.1 * SR)]
    s = Script().add("quiet", silence(0.3)).add("user", command).add("quiet", silence(1.5))
    events, _ = run(people, embedder, vad, s, anchor=anchor, policy="people", people=(ana.id,))
    assert of(events, "rejected")


def run_followup(folks, embedder, vad, script, anchor, speaker, **kw):
    f = Follow(FollowConfig(**kw), ACCURATE, embedder, vad, folks, anchor=anchor, speaker=speaker)
    events = []
    for c in chunks(script.audio):
        events += f.feed(c)
    events += f.finish()
    return events, f


def test_a_one_word_follow_up_goes_through(household, speakers, voice, vad, embedder):
    """Asked a follow-up, Ana says "yes": half a second, too little to tell who
    said it, but the conversation already knows it is her."""
    people, ana, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(1.0)).add("user", user[9 * SR:int(9.5 * SR)]).add("quiet", silence(1.5))
    events, f = run_followup(people, embedder, vad, s, user[:SR // 2], ana.id, policy="people", people=(ana.id,))
    assert not of(events, "rejected")
    assert of(events, "speaker")[0]["id"] == ana.id and of(events, "speaker")[0]["by"] == "conversation"
    assert s.released_from("user", f.released) >= 0.3


def test_a_follow_up_still_leaves_another_voice_out(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("tv", tv[:3 * SR]).add("user", user[9 * SR:11 * SR]).add("quiet", silence(1.5))
    events, f = run_followup(people, embedder, vad, s, None, ana.id, policy="people", people=(ana.id,))
    assert not of(events, "rejected")
    assert s.released_from("tv", f.released) <= 0.5
    assert s.speech_share("user", f.released, vad) >= 0.7


def test_a_short_wake_word_on_a_strict_device_follows_the_voiceprint(household, speakers, voice, vad, embedder):
    """Only Ana gets an answer, and her wake word was too short to say it was
    her: her command is still hers, followed by her voiceprint."""
    people, ana, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.3)).add("user", user[9 * SR:12 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=user[2 * SR:int(2.6 * SR)], policy="people",
                    people=(ana.id,))
    assert next(t for t in f.timeline if t["kind"] == "anchor")["score"] < ACCURATE.threshold
    assert of(events, "speaker")[-1]["id"] == ana.id and not of(events, "rejected")
    assert s.speech_share("user", f.released, vad) >= 0.8


def test_a_guest_on_a_strict_device_still_gets_no_answer(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    guest = held_out(speakers[2])
    s = Script().add("quiet", silence(0.3)).add("user", guest[2 * SR:5 * SR]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=guest[9 * SR:int(9.6 * SR)], policy="people", people=(ana.id,))
    assert not any(e["id"] == ana.id for e in of(events, "speaker"))
    assert of(events, "rejected") or s.released_from("user", f.released) <= 0.5


def test_a_tv_around_a_short_command_does_not_turn_the_user_away(household, speakers, voice, vad, embedder):
    """Only Ana gets an answer, her wake word was too short to say it was her
    and the TV talks before and right after her command. The TV is not who she
    is: her command, with the wake word in front, is."""
    people, ana, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("tv", tv[:int(1.5 * SR)]).add("quiet", silence(0.3))
         .add("user", user[9 * SR:int(10.5 * SR)]).add("tv", tv[2 * SR:5 * SR]).add("quiet", silence(1.5)))
    events, f = run(people, embedder, vad, s, anchor=user[2 * SR:int(2.6 * SR)], policy="people",
                    people=(ana.id,))
    assert next(t for t in f.timeline if t["kind"] == "anchor")["score"] < ACCURATE.threshold
    assert not of(events, "rejected")
    assert of(events, "speaker")[-1]["id"] == ana.id
    assert s.released_from("tv", f.released) <= 0.5


def test_a_guest_with_the_tv_on_is_still_turned_away(household, speakers, voice, vad, embedder):
    people, ana, _ = household
    guest, tv = held_out(speakers[2]), voice[speakers[1]]["test"]
    s = (Script().add("tv", tv[:int(1.5 * SR)]).add("quiet", silence(0.3))
         .add("user", guest[9 * SR:int(10.5 * SR)]).add("tv", tv[2 * SR:5 * SR]).add("quiet", silence(1.5)))
    events, f = run(people, embedder, vad, s, anchor=guest[2 * SR:int(2.6 * SR)], policy="people",
                    people=(ana.id,))
    assert not any(e["id"] == ana.id for e in of(events, "speaker"))
    assert of(events, "rejected") or s.released_from("user", f.released) <= 0.5



@pytest.mark.parametrize("keep", [0.25, 0.0])
def test_the_end_of_the_last_word_is_kept_when_another_voice_starts_right_after(
        household, speakers, voice, vad, embedder, keep):
    """No pause between the user and the TV: the check straddling the two is
    mostly the TV and scores low, so the labels turn to the TV before the
    user's word ends. That much of the TV's labels still goes out as the
    user's. Here the labels are forced to turn at a known point."""
    people, _, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = Script().add("quiet", silence(0.3)).add("user", user[2 * SR:5 * SR]).add("tv", tv[:3 * SR]).add("quiet", silence(1.5))
    # A point inside continuous speech, around where the TV takes over.
    speech = [p >= 0.5 for p in vad.stream().feed(s.audio)]
    tv_start = next(a for who, a, b in s.spans if who == "tv") // FRAME
    k = next(i for i in range(tv_start - 10, len(speech)) if all(speech[i - 6:i + 16]))
    turn = k * FRAME
    f = Follow(FollowConfig(tail_keep_s=keep), ACCURATE, embedder, vad, people, anchor=anchor_of(user))
    f._label = lambda frame, final: "user" if (frame + 1) * FRAME <= turn else "other"
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    end = max(b for a, b in f.released)
    assert end == pytest.approx(turn + int(keep * SR), abs=FRAME)
    assert of(events, "other_voice")[0]["t"] == pytest.approx(end / SR, abs=0.02)


def test_a_later_turn_needs_a_clearer_voice_than_staying_in_one(household, speakers, voice, vad, embedder):
    """Just over the other-voice line keeps a turn going, but once the user is
    done it doesn't start another: a TV scoring there would stop a realtime
    answer. Here the check scores are set by hand."""
    people, _, _ = household
    user = held_out(speakers[0])
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=anchor_of(user))
    for c in chunks(user[:2 * SR]):
        f.feed(c)
    frame = f.n // FRAME - 2
    center = frame * FRAME + FRAME // 2
    other, start = ACCURATE.other_threshold, ACCURATE.other_threshold + 0.5 * (ACCURATE.threshold - ACCURATE.other_threshold)

    def label(score, in_turn, turns):
        f._checks = [(center, score)]
        f.in_turn, f.turns = in_turn, turns
        return f._label(frame, final=True)

    assert label(other + 0.01, in_turn=True, turns=1) == "user"  # inside a turn
    assert label(other + 0.01, in_turn=False, turns=0) == "user"  # the first turn
    assert label(other + 0.01, in_turn=False, turns=1) == "other"  # a later one
    assert label(start + 0.01, in_turn=False, turns=1) == "user"


class Counted:
    """The speaker model, with the length of everything it was asked to embed."""

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self.lengths: list[int] = []

    def embed(self, audio: np.ndarray) -> np.ndarray:
        self.lengths.append(len(audio))
        return self._embedder.embed(audio)


def test_a_user_known_from_the_wake_word_is_not_embedded_again_when_the_turn_ends(
        household, speakers, voice, vad, embedder):
    """The whole command was embedded at the end of the first turn and the
    result thrown away, which held the end of the turn back."""
    people, ana, _ = household
    user = held_out(speakers[0])
    s = Script().add("user", user[2 * SR:7 * SR]).add("quiet", silence(1.5))
    counted = Counted(embedder)
    events, f = run(people, counted, vad, s, anchor=anchor_of(user))
    assert of(events, "speaker")[0]["id"] == ana.id and of(events, "speaker")[0]["by"] == "wake_word"
    assert len(of(events, "turn_end")) == 1
    # The wake word, then only the 1 s voice checks.
    assert all(n == SR for n in counted.lengths[1:])


def test_a_guest_is_still_told_apart_from_the_whole_command(household, speakers, voice, vad, embedder):
    """Someone the wake word didn't recognize is still compared with everyone
    enrolled once their first command is in."""
    people, _, _ = household
    guest = voice[speakers[2]]["test"]
    s = Script().add("user", guest[SR:int(2.6 * SR)]).add("quiet", silence(1.5))
    counted = Counted(embedder)
    events, f = run(people, counted, vad, s, anchor=anchor_of(guest[5 * SR:]))
    assert of(events, "speaker")[-1]["id"] is None
    assert any(t["kind"] == "identify" for t in f.timeline)
    assert any(n > SR for n in counted.lengths[1:])


# ── the end of the wake word ─────────────────────────────────────────────

def wake_end(seconds: float | None, rule: str = "next sound"):
    """Stands in for the Wake Word Aligner: the wake word ends ``seconds``
    into the stream or isn't found (None). Counts how often it was asked."""
    from ground_control.aligner import WakeWordEnd

    def find(anchor, head):
        find.calls.append((len(anchor), len(head)))
        return None if seconds is None else WakeWordEnd(samples=int(seconds * SR), fit=-0.5, rule=rule,
                                                        phonemes="ɐlɛksə", ms=3)
    find.calls = []
    return find


def run_after_wake_word(folks, embedder, vad, audio, anchor, find, **kw):
    f = Follow(FollowConfig(**kw), ACCURATE, embedder, vad, folks, anchor=anchor, wake_word_end=find)
    events = []
    for c in chunks(audio):
        events += f.feed(c)
    events += f.finish()
    return events, f


def talking(audio: np.ndarray, vad) -> np.ndarray:
    """``audio`` from its first speech on, so it starts mid-word like a stream
    that begins before the wake word is over."""
    probs = vad.stream().feed(audio)
    first = next(i for i, p in enumerate(probs) if p >= 0.5)
    return audio[first * FRAME:]


def test_the_end_of_the_wake_word_is_left_out_of_a_command_said_in_one_breath(
        household, speakers, voice, vad, embedder):
    people, ana, _ = household
    user = talking(voice[speakers[0]]["test"], vad)
    audio = np.concatenate([user[:4 * SR], silence(1.5)])
    find = wake_end(0.30)
    events, f = run_after_wake_word(people, embedder, vad, audio, anchor_of(held_out(speakers[0])), find)
    cut = int(0.30 * SR)
    # Asked once, with the wake clip and the first 0.8 s of the stream.
    assert find.calls == [(int(1.2 * SR), int(0.8 * SR))]
    # Nothing before the end of the wake word went out and the command starts right there.
    assert f.released[0][0] == cut
    assert of(events, "turn_start")[0]["t"] == 0.3
    # Half a second of silence in front of it, then the command, faded in.
    sent = f.cleaned_audio()
    lead = int(0.5 * SR)
    assert f.lead_in_s == 0.5 and not sent[:lead].any()
    assert np.allclose(sent[lead + 240:lead + 2000], audio[cut + 240:cut + 2000])
    assert abs(sent[lead]) < 1e-6 and np.abs(sent[lead:lead + 240]).max() <= np.abs(audio[cut:cut + 240]).max()
    assert len(sent) == lead + sum(b - a for a, b in f.released)
    stats = of(events, "stats")[0]
    assert stats["wake_word_s"] == 0.3 and stats["speaker"]["name"] == "Ana"
    assert [t for t in f.timeline if t["kind"] == "wake_word"] == [
        {"t": 0.3, "kind": "wake_word", "found": True, "fit": -0.5, "rule": "next sound", "phonemes": "ɐlɛksə", "ms": 3}]


def test_a_command_after_a_pause_gets_no_silence_in_front(household, speakers, voice, vad, embedder):
    people, *_ = household
    user = talking(voice[speakers[0]]["test"], vad)
    # The end of the wake word, a pause, then the command.
    audio = np.concatenate([user[:int(0.25 * SR)], silence(1.2), user[:4 * SR], silence(1.5)])
    events, f = run_after_wake_word(people, embedder, vad, audio, anchor_of(held_out(speakers[0])),
                                    wake_end(0.30, "pause"))
    # The command has the room's own quiet in front: it goes out as it is.
    assert f.released[0][0] >= int(1.2 * SR)
    assert f.lead_in_s == 0.0
    assert len(f.cleaned_audio()) == sum(b - a for a, b in f.released)
    assert of(events, "stats")[0]["wake_word_s"] == 0.3


def test_a_wake_word_that_isnt_found_changes_nothing(household, speakers, voice, vad, embedder):
    people, *_ = household
    user = talking(voice[speakers[0]]["test"], vad)
    audio = np.concatenate([user[:4 * SR], silence(1.5)])
    anchor = anchor_of(held_out(speakers[0]))
    script = Script().add("user", audio)
    plain, f0 = run(people, embedder, vad, script, anchor=anchor)
    events, f = run_after_wake_word(people, embedder, vad, audio, anchor, wake_end(None))
    assert f.released == f0.released and f.lead_in_s == 0.0
    assert np.array_equal(f.cleaned_audio(), f0.cleaned_audio())
    assert "wake_word_s" not in of(events, "stats")[0]
    assert [t for t in f.timeline if t["kind"] == "wake_word"] == [{"t": 0.0, "kind": "wake_word", "found": False}]


def test_a_wake_word_over_before_the_stream_started_cuts_nothing(household, speakers, voice, vad, embedder):
    people, *_ = household
    user = talking(voice[speakers[0]]["test"], vad)
    audio = np.concatenate([user[:4 * SR], silence(1.5)])
    _, f = run_after_wake_word(people, embedder, vad, audio, anchor_of(held_out(speakers[0])), wake_end(0.0))
    assert f.released[0][0] == 0 and f.lead_in_s == 0.0


def test_a_failure_looking_for_the_wake_words_end_doesnt_stop_the_command(household, speakers, voice, vad, embedder):
    people, *_ = household

    def broken(anchor, head):
        raise RuntimeError("no model")

    audio = np.concatenate([talking(voice[speakers[0]]["test"], vad)[:4 * SR], silence(1.5)])
    events, f = run_after_wake_word(people, embedder, vad, audio, anchor_of(held_out(speakers[0])), broken)
    assert of(events, "turn_end") and f.released[0][0] == 0


def test_the_end_of_the_wake_word_is_looked_for_in_a_stream_shorter_than_the_wait(
        household, speakers, voice, vad, embedder):
    people, *_ = household
    find = wake_end(0.2)
    audio = talking(voice[speakers[0]]["test"], vad)[: int(0.6 * SR)]
    _, f = run_after_wake_word(people, embedder, vad, audio, anchor_of(held_out(speakers[0])), find)
    assert find.calls == [(int(1.2 * SR), len(audio))]
    assert not f.released or f.released[0][0] >= int(0.2 * SR)


def test_without_a_wake_clip_the_end_of_the_wake_word_is_not_looked_for(household, speakers, voice, vad, embedder):
    people, *_ = household
    find = wake_end(0.3)
    audio = np.concatenate([voice[speakers[0]]["test"][:4 * SR], silence(1.5)])
    run_after_wake_word(people, embedder, vad, audio, None, find)
    assert find.calls == []


# ── an answer with no wake word ──────────────────────────────────────────

def test_the_line_for_an_answer_rises_with_how_much_was_said(household, vad, embedder):
    """A short answer needs a score just over the other-voice line, a seventh
    of the way to the recognition threshold. More of it needs more, up to
    the threshold at two seconds."""
    people, *_ = household
    f = Follow(FollowConfig(policy="enrolled"), ACCURATE, embedder, vad, people)
    start = ACCURATE.other_threshold + (ACCURATE.threshold - ACCURATE.other_threshold) / 7
    assert f.answer_line(0.3) == f.answer_line(0.75) == pytest.approx(start) == pytest.approx(0.32)
    assert start < f.answer_line(1.0) < f.answer_line(1.5) < ACCURATE.threshold
    assert f.answer_line(2.0) == f.answer_line(5.0) == ACCURATE.threshold


def test_a_short_answer_with_no_wake_word_is_taken_on_a_device_that_answers_chosen_people(
        household, speakers, voice, vad, embedder):
    """Home Assistant asked a question (ask_question, start_conversation), so
    there is no wake word and "yes" is too short to recognize anyone (#20).
    It sounds like the person the device answers: it goes on, without a name.

    The recognition threshold is set out of reach here, as it is for a real
    "yes" from across a room: these clean recordings would be recognized."""
    from dataclasses import replace

    people, ana, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.5)).add("user", user[2 * SR:int(2.7 * SR)]).add("quiet", silence(1.5))
    f = Follow(FollowConfig(policy="people", people=(ana.id,)), replace(ACCURATE, threshold=0.95), embedder, vad, people)
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    assert not of(events, "rejected") and len(of(events, "turn_end")) == 1
    assert s.speech_share("user", f.released, vad) >= 0.5
    speaker = of(events, "speaker")[-1]
    assert speaker["name"] is None and speaker["by"] == "answer"
    assert speaker["score"] >= speaker["line"] == round(f.answer_line(0.5), 3)
    took = [t for t in f.timeline if t.get("answer")][0]
    assert took["id"] == ana.id and took["speech_s"] < 1.0
    stats = of(events, "stats")[0]
    assert not stats["rejected"] and stats["speaker"]["by"] == "answer"


def test_a_reply_that_reaches_the_threshold_with_the_room_around_it_is_recognized_by_name(
        household, speakers, voice, vad, embedder):
    """A reply's speech frames alone are too little to recognize anyone until
    there are two seconds of them. Scored with the room around it, a reply
    that reaches the recognition threshold is someone recognized, not a short
    answer taken without a name."""
    people, ana, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.5)).add("user", user[2 * SR:int(2.7 * SR)]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, policy="people", people=(ana.id,))
    speaker = of(events, "speaker")[-1]
    assert speaker["name"] == "Ana" and speaker["by"] == "voice" and speaker["score"] >= ACCURATE.threshold
    # It was the reply's own path that named her: its speech alone is under a second.
    assert [t for t in f.timeline if t.get("answer")][0]["speech_s"] < 1.0
    assert not of(events, "rejected")


def test_a_reply_recognized_as_someone_the_device_doesnt_answer_is_turned_away(
        household, speakers, voice, vad, embedder):
    people, ana, leo = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.5)).add("user", user[2 * SR:int(2.7 * SR)]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, policy="people", people=(leo.id,), short_answers=True)
    assert not [e for e in of(events, "speaker") if e["by"] == "answer"]
    assert s.released_from("user", f.released) <= 0.3 or of(events, "rejected")


def test_a_short_answer_from_someone_the_device_doesnt_answer_is_still_turned_away(
        tmp_path, speakers, voice, vad, embedder):
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    guest = held_out(speakers[2])
    s = Script().add("quiet", silence(0.5)).add("user", guest[2 * SR:int(2.7 * SR)]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, policy="people", people=(ana.id,))
    assert not [e for e in of(events, "speaker") if e["by"] == "answer"]
    assert s.released_from("user", f.released) <= 0.3 or of(events, "rejected")


def test_a_short_answer_after_a_wake_word_is_judged_with_the_wake_word(household, speakers, voice, vad, embedder):
    """The line for an answer is only for a session with no wake word: with
    one, a short command is recognized from the wake word and the command."""
    people, ana, _ = household
    user = held_out(speakers[0])
    s = Script().add("user", user[2 * SR:int(2.7 * SR)]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(user), policy="people", people=(ana.id,))
    assert not [t for t in f.timeline if t.get("answer")]
    assert of(events, "speaker")[-1]["name"] == "Ana"


def test_a_click_before_an_answer_is_not_taken_for_the_user(tmp_path, speakers, voice, vad, embedder):
    """A device that answers anyone, no wake word and one 32 ms frame of
    speech long before the answer. The speaker model can't take audio that
    short, which ended the session. A click is nobody's voice anyway."""
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    guest = held_out(speakers[2])
    probs = vad.stream().feed(guest)
    first = next(i for i, p in enumerate(probs) if p >= 0.5) * FRAME
    click = guest[first + 3200:first + 3200 + int(0.04 * SR)]
    s = (Script().add("quiet", silence(0.5)).add("click", click).add("quiet", silence(1.5))
         .add("user", guest[2 * SR:int(2.9 * SR)]).add("quiet", silence(1.5)))
    events, f = run(people, embedder, vad, s)
    assert len(of(events, "turn_end")) == 1
    assert s.speech_share("user", f.released, vad) >= 0.5


def test_the_speaker_model_takes_audio_of_any_length(embedder):
    for n in (0, 160, 512, 1024, 1599):
        e = embedder.embed(np.zeros(n, dtype=np.float32) + 0.01)
        assert e.shape == (embedder.dim,) and np.isfinite(e).all()


def test_a_short_answer_is_only_taken_for_a_turn_that_can_be_one(household, speakers, voice, vad, embedder):
    """The kiosk says how the turn started. A reply to something Home
    Assistant asked can be a short answer. A turn started by hand can't, and
    neither can any turn with short answers turned off in the settings."""
    from dataclasses import replace

    people, ana, _ = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.5)).add("user", user[2 * SR:int(2.7 * SR)]).add("quiet", silence(1.5))
    # The recognition threshold out of reach, as for a real "yes" from across
    # a room: these clean recordings would be recognized by name either way.
    far = replace(ACCURATE, threshold=0.95)

    def play(**kw):
        f = Follow(FollowConfig(policy="people", people=(ana.id,), **kw), far, embedder, vad, people)
        events = []
        for c in chunks(s.audio):
            events += f.feed(c)
        return events + f.finish(), f

    events, f = play(short_answers=False)
    assert of(events, "rejected") and not [t for t in f.timeline if t.get("answer")]
    events, f = play(short_answers=True)
    assert not of(events, "rejected") and of(events, "speaker")[-1]["by"] == "answer"
    # Recognized by name at the usual threshold, short answers or not.
    events, f = run(people, embedder, vad, s, policy="people", people=(ana.id,), short_answers=False)
    assert not of(events, "rejected") and of(events, "speaker")[-1]["name"] == "Ana"


def test_a_short_command_recognition_misses_is_looked_at_again_with_the_room_around_it(
        household, speakers, voice, vad, embedder):
    """Recognition works on the speech frames of the wake word and the
    command put together. For a short command that can fall short of the
    threshold where the command as it was heard doesn't, and a device that
    answers chosen people turned its own person away. Before that, the turn
    is scored once more with the room around it."""
    people, ana, _ = household
    user = held_out(speakers[0])
    # A wake word under the threshold on its own, then a command under a second.
    anchor, command = user[9 * SR:10 * SR], user[int(10.2 * SR):int(11.1 * SR)]
    s = Script().add("quiet", silence(0.4)).add("user", command).add("quiet", silence(1.5))
    f = Follow(FollowConfig(policy="people", people=(ana.id,)), ACCURATE, embedder, vad, people, anchor=anchor)
    # The speech frames say nothing here, as when they fall short.
    noise = np.random.default_rng(0).standard_normal(int(1.5 * SR)).astype(np.float32) * 0.01
    f._identity_audio = lambda last_s=4.0: noise
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    assert not of(events, "rejected")
    speaker = of(events, "speaker")[-1]
    assert speaker["name"] == "Ana" and speaker["by"] == "voice"
    looks = [t for t in f.timeline if t.get("room")]
    assert len(looks) == 1 and "answer" not in looks[0] and looks[0]["score"] >= ACCURATE.threshold


def test_a_second_look_doesnt_let_a_guest_through_after_a_wake_word(tmp_path, speakers, voice, vad, embedder):
    """With a wake word there is no line of its own for a short answer: the
    second look only ever recognizes someone at the threshold."""
    people = People(tmp_path, vad)
    ana = people.add("Ana")
    people.enroll(ana.id, voice[speakers[0]]["enroll"], {"accurate": embedder})
    guest = held_out(speakers[2])
    s = Script().add("quiet", silence(0.4)).add("user", guest[2 * SR:int(2.7 * SR)]).add("quiet", silence(1.5))
    events, f = run(people, embedder, vad, s, anchor=anchor_of(guest), policy="people", people=(ana.id,))
    assert of(events, "rejected") or s.released_from("user", f.released) <= 0.3
    assert not [e for e in of(events, "speaker") if e["name"]]
    assert not [t for t in f.timeline if t.get("answer")]


# ── a voice the face at the kiosk confirms ───────────────────────────────

def _close_voice(household, speakers, face):
    """Ana's command, scored under a recognition threshold set out of reach
    (as for a voice from across the room), with ``face`` at the kiosk."""
    from dataclasses import replace

    people, ana, leo = household
    user = held_out(speakers[0])
    s = Script().add("quiet", silence(0.5)).add("user", user[2 * SR:int(4.5 * SR)]).add("quiet", silence(1.5))
    who = {"ana": ana.id, "leo": leo.id, None: None}[face]
    return s, replace(ACCURATE, threshold=0.95), (lambda: who), people


def test_a_close_voice_is_taken_for_the_person_the_face_says_is_at_the_kiosk(
        household, speakers, voice, vad, embedder):
    s, model, face, people = _close_voice(household, speakers, "ana")
    f = Follow(FollowConfig(policy="enrolled", face=face), model, embedder, vad, people)
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    speaker = of(events, "speaker")[-1]
    assert speaker["name"] == "Ana" and speaker["by"] == "face" and speaker["score"] < model.threshold
    assert [t for t in f.timeline if t["kind"] == "face"][0]["line"] < speaker["score"]
    assert not of(events, "rejected")


def test_a_face_never_names_a_voice_closer_to_someone_else(household, speakers, voice, vad, embedder):
    for face in ("leo", None):
        s, model, at_kiosk, people = _close_voice(household, speakers, face)
        f = Follow(FollowConfig(policy="enrolled", face=at_kiosk), model, embedder, vad, people)
        events = []
        for c in chunks(s.audio):
            events += f.feed(c)
        events += f.finish()
        assert not [e for e in of(events, "speaker") if e["by"] == "face" or e["name"]]
        assert not [t for t in f.timeline if t["kind"] == "face"]


# ── the last sounds of the wake word ──────────────────────────────────────

def ends_at(seconds: float):
    """Stands in for ``Aligner.find``: the wake word ends this far into the stream."""
    from ground_control.aligner import WakeWordEnd

    return lambda anchor, head: WakeWordEnd(samples=int(seconds * SR), fit=-1.2, rule="pause", phonemes="heɪdʒɑːɹvɪs", ms=4)


def test_the_last_sounds_of_the_wake_word_are_not_the_users_speech(household, speakers, voice, vad, embedder):
    """Talk right after the wake word: the stream starts before the word is
    over. The voice detector holds on a little past the word's end, and that
    was taken for a turn. A user who paused before the command had that turn
    end with nothing in it, and a kiosk that answers chosen people rejected
    the session."""
    people, _, _ = household
    guest = held_out(speakers[2])
    # The end of the wake word, a pause of over a second and then the command.
    s = (Script().add("wake", guest[int(1.0 * SR):int(1.25 * SR)]).add("quiet", silence(1.3))
         .add("user", guest[2 * SR:5 * SR]).add("quiet", silence(1.5)))
    f = Follow(FollowConfig(policy="anyone"), ACCURATE, embedder, vad, people, anchor=anchor_of(guest),
               wake_word_end=ends_at(0.15))
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    turns = of(events, "turn_start")
    # One turn, the command: nothing started at the wake word's end.
    assert len(turns) == 1 and turns[0]["t"] > 1.0
    assert s.released_from("wake", f.released) == 0
    assert s.speech_share("user", f.released, vad) >= 0.7
    assert not of(events, "rejected")


def test_a_command_said_in_one_breath_with_the_wake_word_starts_at_its_end(household, speakers, voice, vad, embedder):
    people, _, _ = household
    guest = held_out(speakers[2])
    # No pause: the speech runs on from the wake word into the command.
    s = Script().add("user", guest[2 * SR:5 * SR]).add("quiet", silence(1.5))
    f = Follow(FollowConfig(policy="anyone"), ACCURATE, embedder, vad, people, anchor=anchor_of(guest),
               wake_word_end=ends_at(0.15))
    events = []
    for c in chunks(s.audio):
        events += f.feed(c)
    events += f.finish()
    assert of(events, "turn_start")[0]["t"] <= 0.3 and f.released[0][0] == int(0.15 * SR)
    assert not of(events, "rejected")


def test_a_later_turn_is_compared_with_the_voices_of_its_own_session(household, speakers, voice, vad, embedder):
    """Next to a voice a little like the user's, the voiceprint alone can let
    that voice start a turn. Once a session has heard another voice, speech
    that would start a later turn is compared with that voice and with the
    user as the session heard them, and starts no turn when it is closer to
    the other one."""
    people, _, _ = household
    user, tv = held_out(speakers[0]), voice[speakers[2]]["test"]
    s = (Script().add("quiet", silence(0.3)).add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.0))
         .add("tv", tv[:4 * SR]).add("quiet", silence(1.0)).add("user", user[6 * SR:8 * SR]).add("quiet", silence(1.5)))
    on = FollowConfig(session_voices=True)
    f = Follow(on, ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)])
    events = [e for c in chunks(s.audio) for e in f.feed(c)] + f.finish()
    stats = of(events, "stats")[0]
    # The user's second turn started, after being compared: the TV had been heard by then.
    assert stats["turns"] == 2 and "kept_out" not in stats
    compared = [p for p in f.timeline if p["kind"] == "voices"]
    assert compared and not compared[-1]["kept_out"] and compared[-1]["user"] > compared[-1]["other"] + 0.2
    assert s.released_from("tv", f.released) < 0.3

    # The TV's own speech, asked the same question: closer to the other voice. Its last second is
    # compared with the checks of its first seconds.
    tv_span = next((a, b) for who, a, b in s.spans if who == "tv")
    at = min((c for c, _ in f._checks if tv_span[1] - SR <= c <= tv_span[1] - SR // 2), default=None)
    assert at is not None
    f._start_calls.clear()
    assert f._closer_to_other(at, at // FRAME, final=True) is True
    assert f.kept_out == 1 and [p for p in f.timeline if p["kind"] == "voices"][-1]["kept_out"] is True

    # A session that heard no other voice compares nothing.
    quiet = (Script().add("quiet", silence(0.3)).add("user", user[2 * SR:5 * SR]).add("quiet", silence(1.5))
             .add("user", user[6 * SR:8 * SR]).add("quiet", silence(1.5)))
    f = Follow(on, ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)])
    events = [e for c in chunks(quiet.audio) for e in f.feed(c)] + f.finish()
    assert of(events, "stats")[0]["turns"] == 2 and not [p for p in f.timeline if p["kind"] == "voices"]
    assert f._first_turn_emb is None        # and embeds nothing more for it

    # Off, as it is unless asked for, nothing is compared.
    f = Follow(FollowConfig(), ACCURATE, embedder, vad, people, anchor=user[: int(1.2 * SR)])
    events = [e for c in chunks(s.audio) for e in f.feed(c)] + f.finish()
    assert of(events, "stats")[0]["turns"] == 2 and not [p for p in f.timeline if p["kind"] == "voices"]
