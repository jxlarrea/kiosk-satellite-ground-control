"""One interaction: identify the speaker, catch other voices, find the end.

Pure logic with no I/O, so it can be tested on recorded audio. ``feed``
takes 16 kHz float32 audio in any chunk size and returns events. All times
are seconds from the first sample of the session, pre-roll included.

Events:

- ``speech_start``: speech began.
- ``speaker``: who is speaking (``name`` None when unknown), ``score``,
  ``by`` (``voice`` or ``only_person``). Sent as soon as it's known,
  usually about 1 s into the speech, while the person is still talking.
- ``other_voice``: someone else is talking. ``cut`` is where the speaker's
  last word ended.
- ``end_of_speech``: the speaker is done (``reason`` silence, other_voice
  or max_length).
- ``rejected``: the device only takes enrolled voices and this isn't one.
- ``audio``: gated mode only, the audio to forward, with other voices
  replaced by silence. It lags the input by the hold-back.
- ``stats``: from ``finish``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from . import SAMPLE_RATE
from .engine import Embedder
from .models import SpeakerModel
from .people import People
from .vad import FRAME, VadModel

SR = SAMPLE_RATE


@dataclass
class SessionConfig:
    mode: Literal["live", "gated"] = "live"
    policy: Literal["anyone", "enrolled", "people"] = "anyone"
    people: tuple[str, ...] = ()
    """With the people policy, the only people this device answers."""
    hold_back_s: float | None = None
    """Gated mode lag. None: long enough to catch another voice before it is released."""
    eos_silence_s: float = 0.7
    eos_min_speech_s: float = 0.8
    """Silence ends a turn only after this much speech. A blip (the wake chime's
    echo, a cough) or a one word command is left to the backend's own end of
    speech, which is slower but never cuts off a command."""
    max_s: float = 30.0
    speech_threshold: float = 0.5
    speech_release: float = 0.35
    identify_after_s: float = 1.0
    identify_every_s: float = 1.0
    identify_until_s: float = 6.0
    reject_after_s: float = 2.0
    own_reference_s: float = 2.0
    check_window_s: float = 1.0
    check_speech_fraction: float = 0.5


@dataclass
class _Turn:
    speech_start: int | None = None
    speech_run: int = 0
    silence_run: int = 0
    last_speech_end: int = 0
    speech_idx: list[int] = field(default_factory=list)
    """Frame indices that were speech, in this turn."""
    next_identify: float = 0.0
    speaker: dict[str, Any] | None = None
    speaker_row: np.ndarray | None = None
    best: list[tuple[str, float]] = field(default_factory=list)
    identify_embedding: np.ndarray | None = None
    own: np.ndarray | None = None
    next_check: int = 0
    low_streak: int = 0
    first_low_end: int = 0
    last_pass_end: int = 0
    checks: int = 0
    lowest_check: float | None = None
    cut: int | None = None
    ended: str | None = None
    end_at: int | None = None
    release_limit: int | None = None
    """Gated audio stops here: the cut, or what had arrived when silence ended it."""
    rejected: bool = False


class Session:
    def __init__(self, cfg: SessionConfig, model: SpeakerModel, embedder: Embedder,
                 vad: VadModel, people: People) -> None:
        self.cfg = cfg
        self.model = model
        self.embedder = embedder
        self.people = people
        self._vad = vad.stream()
        self._ids, self._rows = people.voiceprints(model.key)
        everyone = people.all()
        self._only = everyone[0] if len(everyone) == 1 and not people.has_voice(everyone[0].id) else None
        self._buf = np.zeros(int(cfg.max_s * SR) + SR, dtype=np.float32)
        self.n = 0
        self._speech: list[bool] = []
        self._released = 0
        self._t = _Turn()
        hold = cfg.hold_back_s
        if hold is None:
            # Two low checks after the other voice fills half the window, plus margin.
            hold = cfg.check_window_s / 2 + 2 * model.cadence_s + 0.3
        self._hold = int(hold * SR)
        self._started = time.monotonic()
        self._cpu = 0.0
        self.turns = 0
        self.timeline: list[dict[str, Any]] = []
        """Every identification and check, for the Sessions page."""
        self._turn_start = 0
        self._silenced: list[list[int | None]] = []
        """Audio left out of the cleaned copy: from a cut or a rejection to the end of its turn."""

    # ── input ─────────────────────────────────────────────────────────────

    def feed(self, audio: np.ndarray) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        room = len(self._buf) - self.n
        if room <= 0:
            return events
        audio = audio[:room]
        self._buf[self.n:self.n + len(audio)] = audio
        self.n += len(audio)
        t0 = time.process_time()
        for p in self._vad.feed(audio):
            self._frame(p, events)
        if self._t.ended is None and self.n >= self.cfg.max_s * SR:
            self._end("max_length", self.n, events)
        self._cpu += time.process_time() - t0
        if self.cfg.mode == "gated":
            self._release(events, final=False)
        return events

    def new_turn(self) -> list[dict[str, Any]]:
        """Start the next turn of a long (realtime) session: who speaks is checked again."""
        events: list[dict[str, Any]] = []
        if self.cfg.mode == "gated":
            self._release(events, final=True)
        for r in self._silenced:
            if r[1] is None:
                r[1] = self.n
        self._t = _Turn()
        self._turn_start = self.n
        self._released = self.n
        self.turns += 1
        return events

    def finish(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.cfg.mode == "gated":
            self._release(events, final=True)
        t = self._t
        events.append({
            "type": "stats",
            "audio_s": round(self.n / SR, 2),
            "speech_s": round(len(t.speech_idx) * FRAME / SR, 2),
            "speaker": t.speaker,
            "end": t.ended,
            "checks": t.checks,
            "lowest_check": None if t.lowest_check is None else round(t.lowest_check, 3),
            "cpu_ms": round(self._cpu * 1000),
            "wall_ms": round((time.monotonic() - self._started) * 1000),
            "model": self.model.key,
        })
        return events

    # ── per VAD frame ─────────────────────────────────────────────────────

    def _frame(self, p: float, events: list[dict[str, Any]]) -> None:
        cfg, t = self.cfg, self._t
        idx = len(self._speech)
        was = self._speech[-1] if self._speech else False
        speech = p >= cfg.speech_threshold or (was and p >= cfg.speech_release)
        self._speech.append(speech)
        end = (idx + 1) * FRAME
        if t.ended is not None:
            return
        if speech:
            t.speech_idx.append(idx)
            t.speech_run += 1
            t.silence_run = 0
            if t.speech_start is None and t.speech_run >= 3:
                t.speech_start = (idx - 2) * FRAME
                events.append({"type": "speech_start", "t": round(t.speech_start / SR, 3)})
                if self._only is not None and t.speaker is None:
                    self._set_speaker(self._only.id, None, "only_person", events, end)
        else:
            if was:
                t.last_speech_end = idx * FRAME
            t.speech_run = 0
            t.silence_run += 1
        if t.speech_start is None:
            return
        speech_s = len(t.speech_idx) * FRAME / SR
        if t.speaker is None or t.speaker.get("name") is None:
            self._maybe_identify(speech_s, end, events)
        if t.rejected:
            return
        if t.own is None and t.speaker_row is None and speech_s >= cfg.own_reference_s:
            t.own = self.embedder.embed(self._speech_audio())
            t.last_pass_end = end
            t.next_check = end
        self._maybe_check(end, events)
        if (t.ended is None and t.silence_run * FRAME >= cfg.eos_silence_s * SR
                and speech_s >= cfg.eos_min_speech_s):
            self._end("silence", t.last_speech_end, events)

    def _speech_audio(self, last_s: float | None = None) -> np.ndarray:
        idx = self._t.speech_idx
        if last_s is not None:
            idx = idx[-int(last_s * SR / FRAME):]
        return np.concatenate([self._buf[i * FRAME:(i + 1) * FRAME] for i in idx])

    # ── who is speaking ───────────────────────────────────────────────────

    def _maybe_identify(self, speech_s: float, end: int, events: list[dict[str, Any]]) -> None:
        cfg, t = self.cfg, self._t
        if speech_s < max(cfg.identify_after_s, t.next_identify) or speech_s > cfg.identify_until_s:
            return
        if not self._ids:
            if cfg.policy == "enrolled":
                pass  # nobody to compare with: the policy can't apply
            return
        t.next_identify = speech_s + cfg.identify_every_s
        emb = self.embedder.embed(self._speech_audio(4.0))
        scores = self._rows @ emb
        order = np.argsort(-scores)
        t.best = [(self._ids[i], float(scores[i])) for i in order[:2]]
        best_id, best_score = t.best[0]
        self.timeline.append({"t": round(end / SR, 3), "kind": "identify", "score": round(best_score, 3),
                              "id": best_id, "turn": self.turns})
        if best_score >= self.model.threshold:
            t.identify_embedding = emb
            t.speaker_row = self._rows[order[0]]
            t.last_pass_end = end
            t.next_check = end
            self._set_speaker(best_id, best_score, "voice", events, end)
            if cfg.policy == "people" and best_id not in cfg.people:
                t.rejected = True
                events.append({"type": "rejected", "t": round(end / SR, 3), "score": round(best_score, 3)})
                self._end("rejected", end, events)
        elif speech_s >= cfg.reject_after_s and t.speaker is None:
            self._set_speaker(None, best_score, None, events, end)
            if cfg.policy in ("enrolled", "people"):
                t.rejected = True
                events.append({"type": "rejected", "t": round(end / SR, 3), "score": round(best_score, 3)})
                self._end("rejected", end, events)

    def _set_speaker(self, person_id: str | None, score: float | None, by: str | None,
                     events: list[dict[str, Any]], at: int | None = None) -> None:
        person = self.people.get(person_id) if person_id else None
        self._t.speaker = {
            "id": person.id if person else None,
            "name": person.name if person else None,
            "color": person.color if person else None,
            "score": None if score is None else round(score, 3),
            "by": by,
        }
        events.append({"type": "speaker", "t": round((self.n if at is None else at) / SR, 3), **self._t.speaker})

    # ── someone else talking ──────────────────────────────────────────────

    def _maybe_check(self, end: int, events: list[dict[str, Any]]) -> None:
        cfg, t = self.cfg, self._t
        ref = t.speaker_row if t.speaker_row is not None else t.own
        if ref is None or t.ended is not None or end < t.next_check:
            return
        t.next_check = end + int(self.model.cadence_s * SR)
        win = int(cfg.check_window_s * SR)
        start = end - win
        if start < (t.speech_start or 0):
            return
        f0, f1 = start // FRAME, end // FRAME
        frames = self._speech[f0:f1]
        if not frames or sum(frames) / len(frames) < cfg.check_speech_fraction:
            return
        score = float(ref @ self.embedder.embed(self._buf[start:end]))
        self.timeline.append({"t": round(end / SR, 3), "kind": "check", "score": round(score, 3),
                              "low": score < self.model.other_threshold, "turn": self.turns})
        t.checks += 1
        t.lowest_check = score if t.lowest_check is None else min(t.lowest_check, score)
        if score >= self.model.other_threshold:
            t.low_streak = 0
            t.last_pass_end = end
            return
        t.low_streak += 1
        if t.low_streak == 1:
            t.first_low_end = end
        if t.low_streak >= 2:
            t.cut = self._cut_point(t.first_low_end - win, t.last_pass_end)
            events.append({"type": "other_voice", "t": round(end / SR, 3), "cut": round(t.cut / SR, 3),
                           "score": round(score, 3)})
            self._end("other_voice", t.cut, events)

    def _cut_point(self, lo: int, hi: int) -> int:
        """The speaker's last speech-to-silence point in [lo, hi], else lo."""
        lo = max(lo, self._t.speech_start or 0)
        for f in range(min(hi, self.n) // FRAME, lo // FRAME, -1):
            if f < len(self._speech) and self._speech[f - 1] and not self._speech[f]:
                return f * FRAME
        return lo

    # ── the end ───────────────────────────────────────────────────────────

    def _end(self, reason: str, at: int, events: list[dict[str, Any]]) -> None:
        t = self._t
        t.ended, t.end_at = reason, at
        t.release_limit = self.n if reason == "silence" else at
        if reason == "other_voice":
            self._silenced.append([at, None])
        elif reason == "rejected":
            self._silenced.append([self._turn_start, None])
        if self.cfg.mode == "gated":
            self._release(events, final=True)
        events.append({"type": "end_of_speech", "t": round(at / SR, 3), "reason": reason})

    def _release(self, events: list[dict[str, Any]], final: bool) -> None:
        """Gated audio: everything up to the hold-back, silence after a cut."""
        t = self._t
        upto = self.n if final else max(0, self.n - self._hold)
        if t.release_limit is not None:
            upto = min(upto, t.release_limit)
        if upto <= self._released:
            return
        chunk = self._buf[self._released:upto].copy()
        if t.cut is not None and t.cut < upto:
            chunk[max(0, t.cut - self._released):] = 0.0
        events.append({"type": "audio", "t": round(self._released / SR, 3), "audio": chunk})
        self._released = upto

    # ── what the Sessions page keeps ──────────────────────────────────────

    def raw_audio(self) -> np.ndarray:
        return self._buf[:self.n].copy()

    def cleaned_audio(self) -> np.ndarray:
        """What the backend got, or would have in gated mode: other voices silenced."""
        out = self._buf[:self.n].copy()
        for start, stop in self._silenced:
            out[start:self.n if stop is None else stop] = 0.0
        return out
