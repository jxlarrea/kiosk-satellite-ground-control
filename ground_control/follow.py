"""Follow mode: pass on only the user's speech, turn by turn.

The kiosk sends everything its microphone hears for a whole interaction:
one Assist command, or a realtime conversation with many turns. The user is
the person who said the wake word. The kiosk sends that second or so of
audio as the anchor, so a TV talking right after the wake word is never
taken for the user. An enrolled person is recognized from it. Anyone else
is followed by the anchor's own voice.

Every stretch of speech is labeled user or other, by a check of the second
of audio centered on it. Only the user's speech goes back to the kiosk, to
forward to the backend in place of the microphone, with long pauses
shortened and other voices left out, so (tv) (user) (tv) (user) (tv)
reaches the backend as (user) (user). A turn starts at the user's first
word and ends once the user has been quiet for ``eos_silence_s``, whatever
the TV does.

Events:

- ``speaker``: who the user is (``name`` None when unknown), ``score``,
  ``by`` (``wake_word`` or ``voice``).
- ``turn_start``: the user started talking. Released audio follows.
- ``audio``: the user's speech to forward, in order.
- ``other_voice``: someone else talked during a turn and was left out.
- ``turn_end``: the user is done. ``t`` is where their last word ended.

With Isolation on, a turn's audio is held while the user talks and
goes out at ``turn_end``, cleaned of the voices under the user's words. The
isolator runs once over the turn's continuous audio, the way it was trained,
and the released stretches are cut from its output.
- ``rejected``: the device only takes enrolled voices and this isn't one.
  Nothing more is released.
- ``stats``: from ``finish``.

With "Talk right after the wake word" on, the kiosk's stream starts the
moment the wake word engine fires, before the word is over, so its end would
go to the backend in front of the command ("xa turn on the lights"). When
the kiosk says so, the end of the wake word is found (``aligner``) and the
user's speech only counts from there.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import SAMPLE_RATE
from .engine import QUANTUM, Embedder
from .models import ISOLATOR_MODEL, SpeakerModel
from .people import People
from .vad import FRAME, VadModel
from .vulkan import Failed as IsolationFailed

_LOG = logging.getLogger(__name__)

SR = SAMPLE_RATE
MAX_AUDIO_EVENT = SR // 2
"""The most audio in one event: what the kiosk forwards as one message to Home Assistant."""


@dataclass
class FollowConfig:
    policy: str = "anyone"
    """anyone, enrolled (anyone enrolled) or people (only ``people``)."""
    people: tuple[str, ...] = ()
    speech_threshold: float = 0.5
    speech_release: float = 0.35
    check_window_s: float = 1.0
    check_speech_fraction: float = 0.4
    eos_silence_s: float = 0.7
    pad_s: float = 0.2
    """Audio kept before the user's first word, so it isn't clipped."""
    isolation_context_s: float = 2.0
    """How much audio from before the turn the Voice Isolator also gets. With
    the TV heard on its own around the user's words, before them and in the
    second after the last one that the end of the turn waits for, it tells
    better what isn't the user: 2 to 2.5 dB more of the TV goes, measured on
    Echo Show 8 sessions."""
    isolation_floor_db: float = -20.0
    """A turn the Voice Isolator returns quieter than this, against what went
    in, isn't used: the turn goes on as Trimming left it. The model answers a
    voiceprint it doesn't hear with near silence, and now and then takes the
    user for absent in a command of about a second. Trimming had already
    judged those words the user's. In saved sessions a turn came back within
    2 dB of what went in, or 28 to 49 dB under it."""
    tail_keep_s: float = 0.25
    """When another voice starts right as the user finishes, with no pause
    between, the check straddling the two is mostly the other voice and
    scores low, so the end of the user's last word would go with it ("outside"
    cut to "outs"). This much of the other voice's labels, right after the
    user's and in the same stretch of speech, still counts as the user's."""
    gap_keep_s: float = 0.15
    """A pause inside a turn is shortened to this, so left out voices leave no long hole.
    Under 0.25 s, Home Assistant's aggressive end of speech: the kiosk sends it only
    this audio, and a breath between two phrases mustn't end the command there. The
    end of the command is Ground Control's (``eos_silence_s``)."""
    identify_after_s: float = 1.0
    reject_after_s: float = 2.0
    reject_late: bool = True
    """A device that answers only enrolled or chosen people, a wake word that
    named nobody but came close to one of them (the other-voice line or
    more), and ``reject_after_s`` of speech since that named nobody either:
    the session isn't turned away there. It is decided when the first turn
    ends, or at ``reject_late_s`` of speech.

    With a realtime assistant the user waits for the ready chime before the
    command, a second and a half or more. A video playing through that wait
    is taken for the user's speech and used the whole ``reject_after_s`` up
    in three of five real sessions that were turned away next to one. In the
    other two the command over the video fell just short (0.46 and 0.498).

    With Isolation the decision is made on what it returns for the turn,
    with the voiceprint of the closest of those people (``_isolated_look``):
    the loudest ``answer_window_s`` of it has to reach the recognition
    threshold. Those five commands scored 0.54 to 0.62 that way and 0.31 to
    0.50 as the kiosk heard them. Of the same video's 18 stretches held as a
    turn, Isolation returned 16 as near silence and the other two scored
    0.43 and 0.47. It runs only for a session that would otherwise be
    turned away, twice at most: where it used to be turned away, which
    names a user who has said the command by then, and at the end."""
    reject_late_s: float = 6.0
    """The most speech a session waits for under ``reject_late``. A TV that
    talks on after a wake word of its own never ends the turn."""
    min_anchor_speech_s: float = 0.3
    max_turn_s: float = 30.0
    keep_s: float = 120.0
    """Raw and released audio kept for the Sessions page."""
    blip_s: float = 0.15
    """Speech shorter than this, alone, counts as noise outside a turn."""
    anchor_slack: float = 0.08
    """The wake word alone is a short sample of a voice: below the other-voice
    line by this much still counts as the user, until their own speech is in."""
    next_turn_share: float = 0.5
    """A turn after the first starts only on a voice this far from the
    other-voice line toward the recognized one (0.395 with Accurate). The TV
    scoring just over the other-voice line once the user is done must not
    start a turn: in a realtime conversation that stops the answer."""
    judge_later_turns: bool = True
    """With Isolation: a turn after the first is held, judged on what
    Isolation returns for it and only then announced and sent, or left out.

    A check is a second of whatever the microphone hears. Next to a video as
    loud as the user and a little like them, a second of the user talking
    over it is mostly the video, and a second of the video alone can pass for
    the user. No line on such a check keeps one out without losing the other.
    Isolation is what separates two voices that talk at once: run over the
    saved sessions of such a test whole, it returned the user's follow-up in
    nearly every one, including those the checks had lost, and little of the
    video.

    So after the first turn a stretch of speech only has to reach the
    other-voice line to be held as a turn that may be the user's
    (``next_turn_share`` no longer applies). When it ends, Isolation runs on
    it as on any turn. Returned as near silence, it is the user's only if
    the turn as a whole reaches the line a turn starts on today: of the
    video's stretches that came back as silence the best scored 0.36, and a
    follow-up of the user's that Isolation took for absent 0.43. Otherwise
    what Isolation returned is compared with the user and with the session's
    other voice (``judge_margin``, ``judge_line``). A turn that fails is
    left out and the kiosk never hears of it: no turn starts, so the
    assistant's answer isn't stopped and no empty turn is ended."""
    judge_line: float = 0.38
    """In a session that heard no other voice, what Isolation's output for a
    later turn has to score against the user: their voiceprint or their
    voice as this session heard it, whichever is closer."""
    judge_margin: float = 0.08
    """In a session that heard another voice, Isolation's output has to be
    this much closer to the user than to that voice. On 42 saved sessions
    next to a video, the 4 turns of the video that Isolation returned whole
    were at most 0.05 closer to the user, and the user's 23 turns in those
    sessions at least 0.10."""
    judge_retry_strong_only: bool = True
    judge_weak_margin: float = -0.06
    """Isolation is the expensive part, so speech that only came in on the
    lower line is not all handed to it. When its checks are closer to the
    session's other voice than to the user by more than this, it is left out
    without running Isolation. Of 35 such stretches of a video, the 24 that
    had another voice to be compared with were closer to it by 0.08 to 0.28.
    The user's three follow-ups that came in on the lower line were at
    +0.15, +0.01 and -0.04. Speech that reaches the line a turn starts on
    today is always handed to Isolation, as it is today."""
    judge_lead_s: float = 0.6
    judge_tail_s: float = 1.0
    """A turn that is judged takes this much of the speech before its first
    check over the line and after its last, whoever the checks took it for.
    Over a loud video the user's "okay, thank you" had one check at 0.38 and
    the next at 0.28, so half the phrase was labeled the video's. Isolation
    takes the video out of what is held, which the checks can't."""
    session_voices: bool = False
    """Off: on a live kiosk it kept the user's own follow-up out once in 8
    conversations, see the end of this text.

    A turn after the first only starts on a voice that is closer to the
    user as heard in this session (the wake word and the first turn) than to
    the other voice heard in it (the checks under the other-voice line since
    the first turn).

    The voiceprint alone can't always tell: everything a kiosk hears shares
    that kiosk's sound, and next to a video whose voice is a little like the
    user's, the user's own clean turn scored 0.47 against their voiceprint
    and 0.46 against the video's voice. Compared with both voices as this
    session heard them, the video's turns were closer to the other voice by
    0.23, 0.13 and 0.06, and the user's were closer to the user: by 0.04
    talking over that video, 0.05 to 0.14 in the same sessions without it
    and 0.13 or more in 50 turns of other sessions (median 0.37).

    It is decided where a turn would start, on its first checks, so nothing
    is taken back afterwards and it doesn't need Isolation. A session that
    heard no other voice is not affected.

    Live, next to that video (24 conversations): it kept the video from
    starting a turn twice, let two fragments of 0.2 and 0.4 s through, and
    once kept out the user's "ok thank you" said over the video (0.49 to the
    user, 0.59 to the other voice): a second of the user talking over a
    voice as loud as theirs is mostly that voice. Losing the user's words is
    the worse mistake, so it stays off until a turn can be judged on what
    Isolation makes of it."""
    session_other_checks: int = 2
    """This many checks of another voice, at least, to compare with."""
    session_other_gap_s: float = 0.5
    """The other voice's checks are those whose middle is at least this long
    before the speech in question. With 3 checks a second or more before it,
    a video that cut in 1.5 s after the user's command had nothing to be
    compared with and took a turn. With 2 checks half a second before it,
    that turn no longer starts, and on 387 real sessions nothing else
    changes."""
    session_start_checks: int = 3
    """The first checks of the speech a turn would start on, taken together:
    one of the video's turns began on a check that was closer to the user
    by 0.13 and was closer to the video on the next two."""
    keepalive_every_s: float = 0.5
    keepalive_s: float = 0.02
    """Home Assistant Cloud's speech-to-text gives up on a stream that gets
    no audio for about 6 s (400 Invalid HTTP request), so it is kept open
    every ``keepalive_every_s`` while nothing else goes out:

    - Before the user's first speech goes out, such as a pause after the wake
      word, a ``keepalive`` event asks the kiosk to send a little silence
      itself (#4). It isn't audio, so the kiosk still knows nobody has talked
      yet: its own no-speech cancel and its microphone fallback keep working.
    - While Isolation holds a turn, this much silence goes out as audio,
      since the turn has started.

    Neither is part of the saved audio."""
    own_after_s: float = 2.0
    """This much of the user's speech rebuilds the reference of a voice that
    isn't enrolled, from what they actually said."""
    household_wait_s: float = 2.5
    """Without a wake word, a device that answers anyone waits this much speech
    for an enrolled voice, then follows the voice it heard."""
    short_answers: bool | None = None
    """Whether a short answer is taken as below. The kiosk says how the turn
    started: a reply to something Home Assistant asked (``ask_question``,
    ``start_conversation``) or a follow-up gets it, a wake word or a turn
    started by hand doesn't. None when the kiosk doesn't say, as Kiosk
    Satellite before 2026.11.31: then every session that came without a wake
    word gets it. False also when it is turned off in the settings."""
    short_answer_s: float = 0.75
    """A reply to a question Home Assistant asked comes with no wake word, so
    there is nothing but the answer to tell who is talking. A "yes" is too
    short to recognize anyone. A device that answers only enrolled or
    chosen people takes such an answer when it sounds like one of them: up to
    this much speech at ``answer_line_share`` of the way from the other-voice
    line to the recognition threshold, rising to the threshold at
    ``reject_after_s``, as a voice's score against its own voiceprint does.
    From there on it is recognized as any command."""
    answer_line_share: float = 1 / 7
    """Where the line for a short answer starts, between the other-voice line
    and the recognition threshold: 0.32 with the Accurate model's 0.29 and
    0.50. Real one-word answers, scored as below, ran from 0.33 to 0.51
    against the voiceprint of the person who said them. At 0.36 one "yes" in
    seven was turned away and another passed by 0.005. A wrong no fails the
    automation that asked, a wrong yes lets someone answer one question."""
    answer_window_s: float = 1.5
    """A short answer is scored on at least this much audio centered on it,
    with the room around it, as the voice checks are. Its speech frames alone
    are too little for the speaker model: a "yes" scored 0.18 against the
    voiceprint of the person who said it and 0.42 this way, while other
    voices stayed where they were."""
    min_answer_s: float = 0.2
    """Less speech than this says nothing about a voice."""
    wake_word_wait_s: float = 0.8
    """With the end of the wake word to find, nothing is decided until this
    much of the stream is in: the end of the word and the start of the
    command are in it. It costs no time, since a frame is only decided once
    the voice check centered on it is in, which is later than this."""
    wake_tail_s: float = 0.2
    """Speech that runs on from the wake word and stops within this of its
    end is still the wake word: the voice detector holds on for a few frames
    after a word. It was 0.03 to 0.13 s on real kiosk sessions, and the
    shortest command said in one breath with the wake word ran 0.54 s past
    it. Counted as the user's speech, it opened a turn of its own. When the
    user then paused before the command, that turn ended with nothing in it
    to recognize anyone by and a kiosk that answers chosen people rejected
    the session."""
    lead_in_s: float = 0.5
    """A command said in one breath with the wake word starts right where the
    word was cut, with none of the quiet a command has in front of it. Speech
    to text hears a word in audio that starts on a sound ("So what time is
    it"), so this much silence goes in front."""
    fade_in_s: float = 0.015
    """The first of the audio after that silence fades in, so the cut doesn't click."""
    face: Callable[[], str | None] | None = None
    """Confirm voices with faces: who Face Recognition says is at the kiosk
    right now, when the kiosk has it on. A voice under the recognition
    threshold is taken for that person when they are also the closest
    voiceprint and the score reaches ``face_line_share``."""
    face_line_share: float = 1 / 7
    """Where the line for a voice the face confirms sits, between the
    other-voice line and the recognition threshold, as for a short answer
    (``answer_line_share``): 0.32 with the Accurate model."""


class Follow:
    def __init__(self, cfg: FollowConfig, model: SpeakerModel, embedder: Embedder,
                 vad: VadModel, people: People, anchor: np.ndarray | None = None,
                 speaker: str | None = None, isolator=None,
                 wake_word_end: Callable[[np.ndarray, np.ndarray], Any] | None = None) -> None:
        """``wake_word_end``: finds the end of the wake word from the anchor
        and the start of the stream (``Aligner.find``), for a kiosk with "Talk
        right after the wake word" on. Without it nothing is looked for."""
        self.cfg = cfg
        self.isolator = isolator
        """Isolation's model (the Voice Isolator), or None. It needs the Accurate model's voiceprints."""
        self.model = model
        self.embedder = embedder
        self.people = people
        self._vad_model = vad
        self._vad = vad.stream()
        self._ids, self._rows = people.voiceprints(model.key)
        # The voices this device answers. Everyone is still identified, so a
        # session shows who was turned away.
        allowed = [i for i, pid in enumerate(self._ids) if self._allowed(pid)]
        self._house_rows = self._rows[allowed] if len(self._ids) else self._rows
        self._win = int(cfg.check_window_s * SR)
        self._cadence = int(model.cadence_s * SR)
        # A speech frame is decided once the check centered on it is in.
        self._hold = self._win // 2 + self._cadence + int(0.1 * SR)

        # Audio, kept for the last minute so checks can look back.
        self._buf = np.zeros(0, dtype=np.float32)
        self._base = 0
        self.n = 0
        self._speech: list[bool] = []
        self._checks: list[tuple[int, float]] = []
        self._check_embs: dict[int, np.ndarray] = {}
        """Each check's embedding by the middle of its window, to compare with the session's voices."""
        self._tentative = False
        """A turn after the first that is held to be judged: the kiosk hasn't been told of it."""
        self._label_score = 0.0
        self._turn_peak = 0.0
        """The best check a turn's speech was labeled the user's on."""
        self._tentative_events: list[dict[str, Any]] = []
        self.left_out = 0
        """Turns after the first that were held, judged on Isolation's output and left out."""
        self._first_turn: tuple[int, int] | None = None
        self._first_turn_emb: np.ndarray | None = None
        """The first turn's audio as one embedding, made the first time a later turn is compared with it."""
        self._first_turn_end = 0
        self._start_calls: dict[int, bool] = {}
        self._kept_out_runs: set[tuple[int, bool]] = set()
        self.kept_out = 0
        """Stretches of speech that started no turn for being closer to the session's other voice."""
        self._next_check = 0
        self._run_scores: dict[int, float] = {}

        # Who the user is.
        self._ref: np.ndarray | None = None
        self._ref_kind: str | None = None
        self._anchor_emb: np.ndarray | None = None
        self._anchor_speech = np.zeros(0, dtype=np.float32)
        """The wake word's speech: it counts toward who the user is, with what they say after."""
        self._identify_at = 0.0
        self._heard: list[tuple[int, int]] = []
        """With a likely reference: every stretch of speech since the wake word, whoever it sounds like."""
        self._heard_at = 0.0
        self.speaker: dict[str, Any] | None = None
        self._identified = False
        self._rejected = False
        self._likely: int | None = None
        """On a device that answers only some voices: the closest of them to a wake word that named nobody."""
        self._wake_close = False
        """Whether that wake word reached the other-voice line against them, see ``reject_late``."""
        self._looks = 0
        self._look_speech = 0
        self._look: tuple[int, int, np.ndarray, np.ndarray] | None = None
        """What Isolation returned for ``_isolated_look``, for the turn to go out with: from, to, voiceprint, audio."""

        # Turns and what was released.
        self._decided = 0
        self.in_turn = False
        self.turns = 0
        self._turn_start = 0
        self._last_user_end = 0
        self._last_other_end = 0
        self._heard_user_end = 0
        """The end of the last frame labeled the user's by its check, not by the tail grace."""
        self._prev_label: str | None = None
        self._pending_gap: list[tuple[int, int]] = []
        self._other_reported = False
        self._user_speech: list[tuple[int, int]] = []
        """The current first turn's user speech, for identification."""
        self._turn_user_s = 0.0
        self.released: list[tuple[int, int]] = []
        """Input sample ranges released, in order."""
        self._held: list[tuple[int, int]] = []
        self._out_at = 0
        """Where in the stream audio last went out, for the keepalive."""
        """With Isolation on: this turn's released ranges, sent when it ends."""
        self.isolated_ms = 0
        self.isolation_skipped = 0
        self.isolation_failed = 0
        """Turns sent as Trimming left them, since the GPU didn't answer (Vulkan image)."""
        """Turns sent as Trimming left them, since Isolation returned near silence."""
        self.dropped_s = 0.0
        self._raw: list[np.ndarray] = []
        self._raw_n = 0
        self._sent: list[np.ndarray] = []
        self._sent_n = 0
        self.timeline: list[dict[str, Any]] = []
        self.events_kept: list[dict[str, Any]] = []
        self._started = time.monotonic()
        self._cpu = 0.0
        self._pending_events: list[dict[str, Any]] = []
        self._household_embs: dict[int, np.ndarray] = {}
        self._household_settled = False
        self.anchor_audio = anchor.astype(np.float32) if anchor is not None else None
        """The wake word as the kiosk sent it, for the session's audio."""
        self._find_end = wake_word_end if anchor is not None and len(anchor) else None
        """Set until the end of the wake word has been looked for."""
        self._wake_end = 0
        """How much of the stream's start is still the wake word, in samples."""
        self._lead_in_due = False
        self.wake_word: dict[str, Any] | None = None
        """What looking for the end of the wake word found, once it has run."""
        self.lead_in_s = 0.0
        """Silence put in front of the command, at the start of the sent audio."""
        if anchor is not None and len(anchor):
            self._take_anchor(anchor.astype(np.float32))
        if speaker and not self._identified:
            self._take_speaker(speaker)
        if self._ref is None and len(self._house_rows):
            # No wake word (a manual start): the user is whoever of the
            # household talks, and a voice nobody enrolled, like a TV, is not.
            self._ref_kind = "household"

    # ── the anchor ────────────────────────────────────────────────────────

    def _take_speaker(self, person_id: str) -> None:
        """A follow-up in a conversation: an earlier turn already recognized
        the user. Their voiceprint is who to follow, so a one-word answer like
        "yes" or "never mind" needs no identifying of its own, and another
        voice is still left out."""
        if person_id not in self._ids:
            return
        self._ref, self._ref_kind = self._rows[self._ids.index(person_id)], "person"
        self._identified = True
        self.timeline.append({"t": 0.0, "kind": "conversation", "id": person_id})
        self._set_speaker(person_id, None, "conversation", self._pending_events, 0)
        if not self._allowed(person_id):
            self._reject(self._pending_events, 0, None)

    def _take_anchor(self, anchor: np.ndarray) -> None:
        probs = self._vad_model.stream().feed(anchor)
        parts = [anchor[i * FRAME:(i + 1) * FRAME] for i, p in enumerate(probs) if p >= self.cfg.speech_threshold]
        if len(parts) * FRAME < self.cfg.min_anchor_speech_s * SR:
            return
        self._anchor_speech = np.concatenate(parts)
        emb = self.embedder.embed(self._anchor_speech)
        self._anchor_emb = emb
        self._ref, self._ref_kind = emb, "anchor"
        if len(self._ids):
            scores = self._rows @ emb
            best = int(np.argmax(scores))
            score = float(scores[best])
            self.timeline.append({"t": 0.0, "kind": "anchor", "score": round(score, 3), "id": self._ids[best]})
            by = self._named_by(scores, best, "wake_word")
            if by:
                self._ref, self._ref_kind = self._rows[best], "person"
                self._identified = True
                self._set_speaker(self._ids[best], score, by, self._pending_events, 0)
                if not self._allowed(self._ids[best]):
                    self._reject(self._pending_events, 0, score)
            elif self._strict:
                self._follow_likely(scores, emb)

    def _follow_likely(self, scores: np.ndarray, emb: np.ndarray) -> None:
        """A device that answers only enrolled or chosen people, and a wake
        word too short to say who: the user is one of them, so follow the
        closest of their voiceprints with the wake word. A short wake word
        alone is a poor reference, and the user's own next sentence can score
        under the other voice threshold against it. Being followed is not
        being answered: identifying the voice still decides that."""
        allowed = [i for i, pid in enumerate(self._ids) if self._allowed(pid)]
        if not allowed:
            return
        likely = max(allowed, key=lambda i: float(scores[i]))
        self._likely = likely
        self._wake_close = float(scores[likely]) >= self.model.other_threshold
        ref = 2 * self._rows[likely] + emb
        self._ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
        self._ref_kind = "likely"

    # ── input ─────────────────────────────────────────────────────────────

    def feed(self, audio: np.ndarray) -> list[dict[str, Any]]:
        events, self._pending_events = self._pending_events, []
        audio = audio.astype(np.float32)
        t0 = time.process_time()
        self._buf = np.concatenate([self._buf, audio])
        self.n += len(audio)
        if self._raw_n < self.cfg.keep_s * SR:
            self._raw.append(audio.copy())
            self._raw_n += len(audio)
        cfg = self.cfg
        for p in self._vad.feed(audio):
            was = self._speech[-1] if self._speech else False
            self._speech.append(p >= cfg.speech_threshold or (was and p >= cfg.speech_release))
            self._maybe_check(len(self._speech) * FRAME)
        if self._find_end is not None and self.n >= int(cfg.wake_word_wait_s * SR):
            self._find_wake_word_end()
        if self._waiting_for_household(False) and any(self._speech):
            last = len(self._speech) - 1 - self._speech[::-1].index(True)
            if self.n - (last + 1) * FRAME >= cfg.eos_silence_s * SR:
                self._settle_household(self.n, final=True)
        self._decide(events, final=False)
        self._keep_alive(events)
        self._trim()
        self._cpu += time.process_time() - t0
        return self._merge(events)

    def _keep_alive(self, events: list[dict[str, Any]]) -> None:
        """Keeps the stream open while it would get nothing: a keepalive event
        before the user's first speech, a little silence while a turn is held."""
        if self._rejected or not (self._held or not self.released):
            return
        every = int(self.cfg.keepalive_every_s * SR)
        if self.n - self._out_at < every:
            return
        self._out_at = self.n
        t = round(self.n / SR, 3)
        if self._held:
            events.append({"type": "audio", "t": t, "keepalive": True,
                           "audio": np.zeros(int(self.cfg.keepalive_s * SR), dtype=np.float32)})
        else:
            events.append({"type": "keepalive", "t": t, "ms": round(self.cfg.keepalive_s * 1000)})

    def finish(self) -> list[dict[str, Any]]:
        events, self._pending_events = self._pending_events, []
        self._decide(events, final=True)
        if self.in_turn and self.turns == 1 and self._held:
            # A held first turn settles who the user is as it ends, see ``_end_turn``.
            self._end_turn(events)
        if self._ref_kind == "likely" and not self._identified and self._heard:
            self._identify_heard(events, self.n, final=True)
        if self.in_turn:
            self._end_turn(events)
        end = ("rejected" if self._rejected else "other_voice" if self.dropped_s > 0
               else "silence" if self.turns else None)
        events.append({
            "type": "stats",
            "mode": "follow",
            "end": end,
            "audio_s": round(self.n / SR, 2),
            "turns": self.turns,
            "released_s": round(sum(b - a for a, b in self.released) / SR, 2),
            "dropped_s": round(self.dropped_s, 2),
            "speaker": self.speaker,
            "rejected": self._rejected,
            "checks": sum(1 for x in self.timeline if x["kind"] == "check"),
            "cpu_ms": round(self._cpu * 1000),
            **({"isolated_ms": self.isolated_ms} if self.isolator is not None else {}),
            **({"isolation_skipped": self.isolation_skipped} if self.isolation_skipped else {}),
            **({"isolation_failed": self.isolation_failed} if self.isolation_failed else {}),
            **({"kept_out": self.kept_out} if self.kept_out else {}),
            **({"left_out": self.left_out} if self.left_out else {}),
            **({"wake_word_s": round(self._wake_end / SR, 2)} if self.wake_word and self.wake_word["found"] else {}),
            "wall_ms": round((time.monotonic() - self._started) * 1000),
            "model": self.model.key,
        })
        return self._merge(events)

    def _audio(self, a: int, b: int) -> np.ndarray:
        return self._buf[max(a - self._base, 0):max(b - self._base, 0)]

    def _trim(self) -> None:
        if len(self._buf) > 60 * SR:
            drop = len(self._buf) - 30 * SR
            self._buf = self._buf[drop:].copy()
            self._base += drop
            keep_from = self._base
            self._checks = [c for c in self._checks if c[0] >= keep_from - self._win]
            self._check_embs = {c: e for c, e in self._check_embs.items() if c >= keep_from - self._win}

    # ── the end of the wake word ──────────────────────────────────────────

    def _find_wake_word_end(self) -> None:
        """Looks for the end of the wake word in the anchor and the start of
        the stream, once. What comes before it in the stream is not the
        user's command."""
        find, self._find_end = self._find_end, None
        head = self._audio(0, int(self.cfg.wake_word_wait_s * SR)).copy()
        try:
            found = find(self.anchor_audio, head)
        except Exception:  # noqa: BLE001 - the command goes on as it would have without this
            _LOG.exception("looking for the end of the wake word failed")
            return
        if found is None:
            self.wake_word = {"found": False}
            self.timeline.append({"t": 0.0, "kind": "wake_word", "found": False})
            return
        self._wake_end = max(0, min(int(found.samples), len(head)))
        self._lead_in_due = self._wake_end > 0
        self.wake_word = {"found": True, "fit": found.fit, "rule": found.rule, "phonemes": found.phonemes,
                          "ms": found.ms}
        self.timeline.append({"t": round(self._wake_end / SR, 3), "kind": "wake_word", **self.wake_word})

    def _after_wake_word(self, chunk: np.ndarray, a: int, events: list[dict[str, Any]]) -> np.ndarray:
        """The first audio to go out, when it starts right where the wake
        word was cut: silence goes out in front of it and it fades in. A
        command after a pause has the room's own quiet in front and goes out
        as it is."""
        if not self._lead_in_due:
            return chunk
        self._lead_in_due = False
        if a != self._wake_end:
            return chunk
        self.lead_in_s = self.cfg.lead_in_s
        self._send(np.zeros(int(self.cfg.lead_in_s * SR), dtype=np.float32), a, events)
        fade = min(len(chunk), int(self.cfg.fade_in_s * SR))
        chunk = chunk.copy()
        chunk[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        return chunk

    # ── who is talking, frame by frame ────────────────────────────────────

    def _maybe_check(self, end: int) -> None:
        if end < self._next_check or end < self._win:
            return
        start = end - self._win
        f0, f1 = start // FRAME, end // FRAME
        frames = self._speech[f0:f1]
        if not frames or sum(frames) / len(frames) < self.cfg.check_speech_fraction:
            return
        self._next_check = end + self._cadence
        emb = self.embedder.embed(self._audio(start, end))
        if self._ref_kind == "household":
            scores = self._house_rows @ emb
            score = float(np.max(scores))
            self._checks.append((end - self._win // 2, score))
            self.timeline.append({"t": round(end / SR, 3), "kind": "check", "score": round(score, 3),
                                  "low": score < self.model.other_threshold, "turn": self.turns})
            self._household_embs[end - self._win // 2] = emb
            if score >= self.model.other_threshold:
                self._household_settled = True
            self._settle_household(end, final=False)
            return
        if self._ref is None:
            # No anchor: the first second of speech is the user.
            if sum(frames) / len(frames) >= 0.6:
                self._ref, self._ref_kind = emb, "first"
            return
        score = float(self._ref @ emb)
        if self._ref_kind == "anchor":
            score += self.cfg.anchor_slack
        self._checks.append((end - self._win // 2, score))
        self._check_embs[end - self._win // 2] = emb
        self.timeline.append({"t": round(end / SR, 3), "kind": "check", "score": round(score, 3),
                              "low": score < self.model.other_threshold, "turn": self.turns})

    def _settle_household(self, at: int, final: bool) -> None:
        """No wake word, a device that answers anyone and nobody enrolled heard
        yet: once enough speech is in, the voice heard is the user."""
        if self._ref_kind != "household" or self._household_settled or self.cfg.policy != "anyone":
            return
        if not final and sum(self._speech) * FRAME < self.cfg.household_wait_s * SR:
            return
        if self._household_embs:
            ref = np.mean(list(self._household_embs.values()), axis=0)
        else:
            # A word too short for a check: its own audio is the reference.
            frames = [i for i, sp in enumerate(self._speech) if sp and i * FRAME >= self._base]
            # A click is not a voice to follow. The speaker model can't take audio that short either.
            if len(frames) * FRAME < self.cfg.blip_s * SR:
                return
            ref = self.embedder.embed(np.concatenate([self._audio(i * FRAME, (i + 1) * FRAME) for i in frames]))
        self._ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
        self._ref_kind = "first"
        self._checks = [(c, float(self._ref @ self._household_embs[c]) if c in self._household_embs else s)
                        for c, s in self._checks]
        self._run_scores = {}
        self._household_embs = {}
        self.timeline.append({"t": round(at / SR, 3), "kind": "reference", "turn": self.turns})

    def _waiting_for_household(self, final: bool) -> bool:
        return (not final and self._ref_kind == "household" and not self._household_settled
                and self.cfg.policy == "anyone")

    def _start_line(self) -> float | None:
        """The score a voice needs to start a turn after the first, or None inside a turn."""
        if self.in_turn or self.turns == 0:
            return None
        m = self.model
        if self._judging():
            return m.other_threshold  # held and judged at its end, see ``judge_later_turns``
        return m.other_threshold + self.cfg.next_turn_share * (m.threshold - m.other_threshold)

    def _judging(self) -> bool:
        """Whether turns after the first are held and judged on Isolation's output."""
        return (self.cfg.judge_later_turns and self.isolator is not None and self.model.key == ISOLATOR_MODEL
                and self._ref is not None and self._ref_kind != "household")

    def _label(self, f: int, final: bool) -> str | None:
        """user or other for speech frame ``f``, or None to wait for its check."""
        fs, fe = f * FRAME, (f + 1) * FRAME
        if not final and self.n - fe < self._hold:
            return None
        if final:
            self._settle_household(self.n, final=True)
        if self._ref is None and self._ref_kind != "household":
            return "user"
        center = (fs + fe) // 2
        near = [c for c in self._checks if abs(c[0] - center) <= self._win * 6 // 10]
        if near:
            at, score = min(near, key=lambda c: abs(c[0] - center))
            if score < self.model.other_threshold and self._waiting_for_household(final):
                return None
            start = self._start_line()
            self._label_score = score
            if score < (start or self.model.other_threshold):
                return "other"
            if start is not None:
                # A turn after the first would start here.
                other = self._closer_to_other(at, f, final)
                if other is None:
                    return None
                if other:
                    return "other"
            return "user"
        # Too little speech around it for a check: a word on its own.
        r0, r1 = self._run(f)
        if (r1 - r0) * FRAME < self.cfg.blip_s * SR:
            return "user" if self.in_turn else "other"
        if r1 >= len(self._speech) and not final:
            return None  # the run is still going
        score = self._run_scores.get(r0)
        if score is None:
            word = self.embedder.embed(self._audio(r0 * FRAME, r1 * FRAME))
            score = float(np.max(self._house_rows @ word)) if self._ref_kind == "household" else float(self._ref @ word)
            if self._ref_kind == "anchor":
                score += self.cfg.anchor_slack
            self._run_scores[r0] = score
            self.timeline.append({"t": round(r1 * FRAME / SR, 3), "kind": "word", "score": round(score, 3),
                                  "turn": self.turns})
        # Lean to keeping a short word: a missed "yes" is worse than a TV word.
        if score < self.model.other_threshold - 0.05 and self._waiting_for_household(final):
            return None
        line = self._start_line() or self.model.other_threshold - 0.05
        return "user" if score >= line else "other"

    def _closer_to_other(self, at: int, f: int, final: bool) -> bool | None:
        """Whether the speech a turn after the first would start on, at the
        check in the middle of ``at``, is closer to the other voice this
        session heard than to its user (``session_voices``). None to wait for
        the checks that follow it. False when there is nothing to compare."""
        cfg = self.cfg
        if at in self._start_calls:
            return self._start_calls[at]
        if not cfg.session_voices or self._first_turn is None:
            return False
        # The other voice: checks under the other-voice line since the first
        # turn, whose windows ended before this one's began.
        other = [self._check_embs[c] for c, s in self._checks
                 if s < self.model.other_threshold and self._first_turn_end < c <= at - int(cfg.session_other_gap_s * SR)
                 and c in self._check_embs]
        if len(other) < cfg.session_other_checks:
            return False
        if self._first_turn_emb is None:
            a, b = self._first_turn
            if a >= self._base and b - a >= SR // 2:
                self._first_turn_emb = self.embedder.embed(self._audio(a, b))
        user = [e for e in (self._anchor_emb, self._first_turn_emb) if e is not None]
        if not user:
            return False
        last = at + (cfg.session_start_checks - 1) * self._cadence + self._cadence // 2
        mine = [self._check_embs[c] for c, _ in self._checks if at <= c <= last and c in self._check_embs]
        if len(mine) < cfg.session_start_checks and not final:
            _, r1 = self._run(f)
            still_talking = r1 >= len(self._speech)
            if still_talking and self.n < last + self._win // 2 + self._cadence:
                return None  # its next checks are still to come
        u = np.sum(user, axis=0)
        o = np.sum(other, axis=0)
        u, o = u / max(float(np.linalg.norm(u)), 1e-9), o / max(float(np.linalg.norm(o)), 1e-9)
        to_user = float(np.mean([u @ e for e in mine]))
        to_other = float(np.mean([o @ e for e in mine]))
        out = to_other > to_user
        self._start_calls[at] = out
        r0, _ = self._run(f)
        key = (r0, out)
        if key not in self._kept_out_runs:
            # Once for a stretch of speech and outcome, not for every frame of it.
            self._kept_out_runs.add(key)
            if out:
                self.kept_out += 1
            self.timeline.append({"t": round(at / SR, 3), "kind": "voices", "user": round(to_user, 3),
                                  "other": round(to_other, 3), "kept_out": out, "turn": self.turns})
        return out

    def _wake_word_tail(self, f: int, final: bool) -> bool | None:
        """Whether the speech at frame ``f`` is still the wake word: a run of
        speech that began before the word's end and stops within
        ``wake_tail_s`` of it. None while the run is still going and could
        be either."""
        if not self._wake_end:
            return False
        r0, r1 = self._run(f)
        if r0 * FRAME >= self._wake_end:
            return False
        if r1 * FRAME - self._wake_end >= self.cfg.wake_tail_s * SR:
            return False  # it goes on: the command, said in one breath
        if r1 >= len(self._speech) and not final:
            return None
        return True

    def _run(self, f: int) -> tuple[int, int]:
        r0 = f
        while r0 > 0 and self._speech[r0 - 1]:
            r0 -= 1
        r1 = f + 1
        while r1 < len(self._speech) and self._speech[r1]:
            r1 += 1
        return r0, r1

    # ── turns and release ─────────────────────────────────────────────────

    def _decide(self, events: list[dict[str, Any]], final: bool) -> None:
        if self._find_end is not None:
            # Where the wake word ends comes first: what is before it is not a command.
            if not final:
                return
            self._find_wake_word_end()
        while self._decided < len(self._speech):
            f = self._decided
            if self._rejected:
                self._decided = len(self._speech)
                return
            if self._speech[f] and (f + 1) * FRAME > self._wake_end:
                tail = self._wake_word_tail(f, final)
                if tail is None:
                    return
                label = "gap" if tail else self._label(f, final)
                if label is None:
                    return
            else:
                label = "gap"
            self._apply(f, label, events)
            self._decided += 1

    def _apply(self, f: int, label: str, events: list[dict[str, Any]]) -> None:
        cfg = self.cfg
        fs, fe = f * FRAME, (f + 1) * FRAME
        # The frame the wake word ends in counts from where it ends.
        fs = max(fs, min(self._wake_end, fe))
        keep = cfg.judge_tail_s if self._tentative else cfg.tail_keep_s
        tail = (label == "other" and self.in_turn and self._prev_label in ("user", "tail")
                and fe - self._heard_user_end <= int(keep * SR))
        self._prev_label = "tail" if tail else label
        if tail:
            # The end of the user's word, not the other voice yet: released, nothing more.
            self._release(fs, fe, events)
            self._last_user_end = fe
            return
        if label == "user":
            self._heard_user_end = fe
            self._turn_peak = max(self._turn_peak, self._label_score)
        if label in ("user", "other") and self._ref_kind == "likely" and not self._identified:
            self._heard.append((fs, fe))
            self._identify_heard(events, fe, final=False)
            if self._rejected:
                return
        if label == "user":
            if not self.in_turn:
                self._tentative = self.turns >= 1 and self._judging()
                self._tentative_events = []
                self._turn_peak = 0.0
                self.in_turn = True
                self.turns += 1
                self._turn_start = fs
                self._other_reported = False
                self._turn_user_s = 0.0
                if not self._tentative:
                    events.append({"type": "turn_start", "t": round(fs / SR, 3), "turn": self.turns})
                if self._tentative:
                    # Held and cleaned by Isolation: the speech before it comes along, whatever it was taken for.
                    pad_from = max(fs - int(cfg.judge_lead_s * SR), self._decided_floor(), self._wake_end, self._base)
                else:
                    pad_from = max(fs - int(cfg.pad_s * SR), self._last_other_end, self._decided_floor(),
                                   self._wake_end)
                if pad_from < fs:
                    self._release(pad_from, fs, events)
            elif self._pending_gap:
                self._release_gap(self._pending_gap, int(cfg.gap_keep_s * SR), events)
            self._pending_gap = []
            self._release(fs, fe, events)
            self._last_user_end = fe
            self._other_reported = False
            self._turn_user_s += FRAME / SR
            if self.turns == 1 or not self._identified or self._ref_kind in ("anchor", "first"):
                self._user_speech.append((fs, fe))
                self._maybe_identify(events, fe)
                self._maybe_own(fe)
        elif label == "other":
            self._last_other_end = fe
            if self.in_turn:
                self.dropped_s += FRAME / SR
                if not self._other_reported:
                    self._other_reported = True
                    (self._tentative_events if self._tentative else events).append(
                        {"type": "other_voice", "t": round(fs / SR, 3), "turn": self.turns})
        else:
            if self.in_turn:
                self._pending_gap.append((fs, fe))
        if self.in_turn:
            quiet = fe - self._last_user_end
            if quiet >= cfg.eos_silence_s * SR or fe - self._turn_start >= cfg.max_turn_s * SR:
                self._end_turn(events)

    def _release_gap(self, gap: list[tuple[int, int]], keep: int, events: list[dict[str, Any]]) -> None:
        """The quiet frames of a pause, shortened to ``keep`` samples: half from
        each end, so neither neighboring word is clipped. Frames of a voice left
        out in the middle are not in ``gap`` and never go."""
        total = sum(b - a for a, b in gap)
        if total <= keep:
            for a, b in gap:
                self._release(a, b, events)
            return
        head, tail = keep // 2, keep - keep // 2
        out: list[tuple[int, int]] = []
        for a, b in gap:
            if head <= 0:
                break
            take = min(b - a, head)
            out.append((a, a + take))
            head -= take
        back: list[tuple[int, int]] = []
        for a, b in reversed(gap):
            if tail <= 0:
                break
            take = min(b - a, tail)
            back.append((b - take, b))
            tail -= take
        for a, b in out + list(reversed(back)):
            self._release(a, b, events)

    def _decided_floor(self) -> int:
        """Audio already passed over as another voice or an earlier turn is not padding."""
        return self.released[-1][1] if self.released else 0

    def _end_turn(self, events: list[dict[str, Any]]) -> None:
        self.in_turn = False
        self._pending_gap = []
        first = self.turns == 1
        if first and self._first_turn is None:
            # The user as this session heard them: what later turns are compared with, see ``_closer_to_other``.
            self._first_turn = (max(self._turn_start - int(self.cfg.pad_s * SR), self._wake_end), self._last_user_end)
            self._first_turn_end = self._last_user_end
        early = first and bool(self._held)
        if early:
            # Isolation keeps the voice of the voiceprint it is given: who the
            # user is gets settled first, so a short command is cleaned with
            # their profile and not the wake word's guess, and a voice turned
            # away sends nothing.
            self._after_first_turn(events)
        tentative, self._tentative = self._tentative, False
        waiting, self._tentative_events = self._tentative_events, []
        mark = len(events)
        if not self._send_held(events, judge=tentative):
            # Judged not to be the user: as if no turn had started.
            self.turns -= 1
            self._last_other_end = max(self._last_other_end, self._last_user_end)
            return
        if tentative:
            events[mark:mark] = [{"type": "turn_start", "t": round(self._turn_start / SR, 3), "turn": self.turns}, *waiting]
        events.append({"type": "turn_end", "t": round(self._last_user_end / SR, 3), "turn": self.turns,
                       "speech_s": round(self._turn_user_s, 2)})
        if first and not early:
            self._after_first_turn(events)

    def _release(self, a: int, b: int, events: list[dict[str, Any]]) -> None:
        if b <= a or self._rejected:
            return
        chunk = self._audio(a, b).copy()
        if not len(chunk):
            return
        if self.released and self.released[-1][1] == a:
            self.released[-1] = (self.released[-1][0], b)
        else:
            self.released.append((a, b))
        if self.isolator is not None and self.in_turn:
            if self._held and self._held[-1][1] == a:
                self._held[-1] = (self._held[-1][0], b)
            else:
                self._held.append((a, b))
            return
        self._send(self._after_wake_word(chunk, a, events), a, events)

    def _send_held(self, events: list[dict[str, Any]], judge: bool = False) -> bool:
        """The turn's held audio, through the Voice Isolator when the user's
        voiceprint is one it knows. With ``judge`` (a turn after the first,
        see ``judge_later_turns``) the turn is also judged on what comes back.
        False when it was left out."""
        held, self._held = self._held, []
        if not held or self._rejected:
            return True
        start, end = held[0][0], held[-1][1]
        span = self._audio(start, end).copy()
        if len(span) < end - start:  # older than the kept audio: can't happen within max_turn_s
            return True
        voiceprint = self._ref if self.model.key == ISOLATOR_MODEL else None
        if judge and voiceprint is not None:
            m_ = self.model
            if self._turn_peak < m_.other_threshold + self.cfg.next_turn_share * (m_.threshold - m_.other_threshold):
                raw = self._raw_scores(start, end)
                if "raw_user" in raw and raw["raw_user"] - raw["raw_other"] < self.cfg.judge_weak_margin:
                    # By its checks already the session's other voice: not worth a run of Isolation.
                    self.left_out += 1
                    self.timeline.append({"t": round(end / SR, 3), "kind": "judged", "judged": True, "left_out": True,
                                          "unheard": True, "peak": round(self._turn_peak, 3), **raw, "turn": self.turns})
                    self._leave_out(held)
                    return False
        if voiceprint is not None:
            t0 = time.perf_counter()
            # The isolator hears everything kept around the turn, up to now.
            ctx = max(self._base, start - int(self.cfg.isolation_context_s * SR))
            look, self._look = self._look, None
            try:
                if look is not None and look[:2] == (ctx, self.n) and look[2] is voiceprint:
                    around = look[3]  # what recognized the user is what goes out
                else:
                    around = self.isolator.clean(self._audio(ctx, self.n).copy(), voiceprint)
            except IsolationFailed as err:
                # The GPU didn't answer (Vulkan image): the turn goes on as Trimming left it.
                _LOG.warning("Isolation not used for this command: %s", err)
                around = None
            ms = round((time.perf_counter() - t0) * 1000)
            self.isolated_ms += ms
            step = {"t": round(end / SR, 3), "kind": "isolated", "ms": ms, "turn": self.turns}
            if around is None:
                self.isolation_failed += 1
                step.update(skipped=True, failed=True)
                self.timeline.append(step)
                voiceprint = None
        if voiceprint is not None:
            cleaned_span = around[start - ctx:end - ctx]
            # How much of the turn came back, over the pieces that go on.
            went_in = np.concatenate([span[a - start:b - start] for a, b in held])
            came_back = np.concatenate([cleaned_span[a - start:b - start] for a, b in held])
            kept_db = 20 * np.log10((float(np.sqrt(np.mean(came_back ** 2))) + 1e-9)
                                    / (float(np.sqrt(np.mean(went_in ** 2))) + 1e-9))
            gone = kept_db < self.cfg.isolation_floor_db
            m_ = self.model
            strong = self._turn_peak >= m_.other_threshold + self.cfg.next_turn_share * (m_.threshold - m_.other_threshold)
            if judge:
                step.update(peak=round(self._turn_peak, 3), **self._raw_scores(start, end))
            if judge and gone and (strong or not self.cfg.judge_retry_strong_only):
                # Isolation answers a lone voice whole or not at all, and which one can turn on how
                # much audio it is given. Before the user's words are given up: once more, on less.
                # Only for speech that would have started a turn by its checks alone: the rest came
                # in on the lower line and Isolation's first answer stands.
                near = max(self._base, start - SR)
                try:
                    again = self.isolator.clean(self._audio(near, min(self.n, end + SR // 2)).copy(), voiceprint)
                except IsolationFailed:
                    again = None
                if again is not None:
                    back = np.concatenate([again[a - near:b - near] for a, b in held])
                    db = 20 * np.log10((float(np.sqrt(np.mean(back ** 2))) + 1e-9)
                                       / (float(np.sqrt(np.mean(went_in ** 2))) + 1e-9))
                    if db >= self.cfg.isolation_floor_db:
                        around, ctx, kept_db, gone = again, near, db, False
                        cleaned_span = around[start - ctx:end - ctx]
                        step.update(again=True)
            if judge:
                if gone:
                    # Isolation hears none of the user in it, as it now and then does with a short command
                    # of theirs: theirs if the turn as a whole reaches the line a turn starts on without
                    # all this. It then goes on as Trimming left it, as such a turn always has.
                    score = self._turn_score(self._buf, self._base, start, end)
                    line = m_.other_threshold + self.cfg.next_turn_share * (m_.threshold - m_.other_threshold)
                else:
                    # What Isolation kept is judged where it kept the most: its loudest second and a half.
                    la, lb = self._loudest(around, ctx, start, end)
                    near_ = self._session_scores(around, ctx, la, lb)
                    step.update(near_)
                    score = max(self._turn_score(around, ctx, la, lb), near_.get("to_user", -1.0))
                    if "to_other" in near_:
                        score, line = score - near_["to_other"], self.cfg.judge_margin
                    else:
                        line = self.cfg.judge_line
                step.update(judged=True, score=round(score, 3), line=round(line, 3))
                if score < line:
                    self.left_out += 1
                    step.update(left_out=True, **({"kept_db": round(kept_db)} if gone else {}))
                    self.timeline.append(step)
                    self._leave_out(held)
                    return False
            if gone:
                # Near silence for words Trimming took for the user's: not used.
                self.isolation_skipped += 1
                step.update(skipped=True, kept_db=round(kept_db))
            else:
                span = cleaned_span
            self.timeline.append(step)
        elif judge:
            # No answer from Isolation to judge it on: the line a later turn starts on without it.
            m = self.model
            score = self._turn_score(self._buf, self._base, start, end)
            line = m.other_threshold + self.cfg.next_turn_share * (m.threshold - m.other_threshold)
            self.timeline.append({"t": round(end / SR, 3), "kind": "judged", "score": round(score, 3), "line": round(line, 3),
                                  "left_out": score < line, "turn": self.turns})
            if score < line:
                self.left_out += 1
                self._leave_out(held)
                return False
        cleaned = self._after_wake_word(np.concatenate([span[a - start:b - start] for a, b in held]), start, events)
        # In pieces no bigger than the stream's own, so a whole turn doesn't
        # reach Home Assistant as one message.
        for i in range(0, len(cleaned), MAX_AUDIO_EVENT):
            self._send(cleaned[i:i + MAX_AUDIO_EVENT], start, events)
        return True

    def _turn_score(self, audio: np.ndarray, base: int, a: int, b: int) -> float:
        """A turn scored as a whole against the voice being followed, with the
        room around it as a short answer is (``_answer_audio``). ``audio`` is
        what is scored, starting at sample ``base``: what the kiosk heard or
        what Isolation made of it."""
        n = min(len(audio), max(int(self.cfg.answer_window_s * SR), -(-(b - a) // QUANTUM) * QUANTUM))
        lo = max(0, min((a + b) // 2 - n // 2 - base, len(audio) - n))
        return float(self._ref @ self.embedder.embed(audio[lo:lo + n]))

    def _raw_scores(self, a: int, b: int) -> dict[str, float]:
        """The checks inside a held turn against the voices of this session, as the kiosk heard them."""
        mine = [self._check_embs[c] for c, _ in self._checks if a <= c <= b and c in self._check_embs]
        other = [self._check_embs[c] for c, sc in self._checks
                 if sc < self.model.other_threshold and self._first_turn_end < c <= a - self._win // 2 and c in self._check_embs]
        if not mine or len(other) < 2:
            return {}
        o = np.sum(other, axis=0)
        o = o / max(float(np.linalg.norm(o)), 1e-9)
        out = {"raw_other": round(float(np.mean([o @ e for e in mine])), 3)}
        u = self._session_user()
        if u is not None:
            out["raw_user"] = round(float(np.mean([u @ e for e in mine])), 3)
        return out

    def _session_user(self) -> np.ndarray | None:
        """The user as this session heard them: the wake word and the first
        turn, one unit-length reference. The first turn is embedded the first
        time it is needed."""
        if self._first_turn is not None and self._first_turn_emb is None:
            fa, fb = self._first_turn
            if fa >= self._base and fb - fa >= SR // 2:
                self._first_turn_emb = self.embedder.embed(self._audio(fa, fb))
        user = [x for x in (self._anchor_emb, self._first_turn_emb) if x is not None]
        if not user:
            return None
        u = np.sum(user, axis=0)
        return u / max(float(np.linalg.norm(u)), 1e-9)

    def _loudest(self, audio: np.ndarray, base: int, a: int, b: int) -> tuple[int, int]:
        """The ``answer_window_s`` of ``audio`` between samples ``a`` and ``b``
        with the most energy: where the voice Isolation kept is. A held turn
        can be several seconds of which the user said one."""
        n = int(self.cfg.answer_window_s * SR)
        if b - a <= n:
            return a, b
        x = audio[a - base:b - base].astype(np.float64) ** 2
        c = np.concatenate([[0.0], np.cumsum(x)])
        hop = SR // 10
        starts = np.arange(0, len(x) - n + 1, hop)
        best = int(starts[int(np.argmax(c[starts + n] - c[starts]))])
        return a + best, a + best + n

    def _session_scores(self, audio: np.ndarray, base: int, a: int, b: int) -> dict[str, float]:
        """What Isolation returned for a turn, against the voices of this
        session: the user as heard in the wake word and first turn, and the
        other voice as heard in the checks under the other-voice line."""
        n = min(len(audio), max(int(self.cfg.answer_window_s * SR), -(-(b - a) // QUANTUM) * QUANTUM))
        lo = max(0, min((a + b) // 2 - n // 2 - base, len(audio) - n))
        e = self.embedder.embed(audio[lo:lo + n])
        out: dict[str, float] = {}
        u = self._session_user()
        if u is not None:
            out["to_user"] = round(float(u @ e), 3)
        other = [self._check_embs[c] for c, sc in self._checks
                 if sc < self.model.other_threshold and self._first_turn_end < c <= a - self._win // 2 and c in self._check_embs]
        if len(other) >= 2:
            o = np.sum(other, axis=0)
            out["to_other"] = round(float(o @ e / max(float(np.linalg.norm(o)), 1e-9)), 3)
        return out

    def _leave_out(self, held: list[tuple[int, int]]) -> None:
        """A held turn that turned out not to be the user's: nothing of it
        goes to the assistant and it no longer counts as released."""
        for a, b in held:
            kept: list[tuple[int, int]] = []
            for ra, rb in self.released:
                if rb <= a or ra >= b:
                    kept.append((ra, rb))
                    continue
                if ra < a:
                    kept.append((ra, a))
                if rb > b:
                    kept.append((b, rb))
            self.released = kept
            self.dropped_s += (b - a) / SR

    def _send(self, chunk: np.ndarray, a: int, events: list[dict[str, Any]]) -> None:
        self._out_at = self.n
        if self._sent_n < self.cfg.keep_s * SR:
            self._sent.append(chunk)
            self._sent_n += len(chunk)
        events.append({"type": "audio", "t": round(a / SR, 3), "audio": chunk})

    # ── who the user is ───────────────────────────────────────────────────

    def _user_audio(self, last_s: float = 4.0) -> np.ndarray:
        parts = [self._audio(a, b) for a, b in self._user_speech]
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return audio[-int(last_s * SR):]

    def _identity_audio(self, last_s: float = 4.0) -> np.ndarray:
        """The user's speech so far with the wake word in front, for a command
        too short on its own to tell who said it."""
        return np.concatenate([self._anchor_speech, self._user_audio(last_s)])[-int(last_s * SR):]

    def _maybe_identify(self, events: list[dict[str, Any]], at: int) -> None:
        if self._identified or not len(self._ids):
            return
        speech_s = sum(b - a for a, b in self._user_speech) / SR
        # Again with every quarter second of new speech, not every frame.
        if speech_s < self.cfg.identify_after_s or speech_s < self._identify_at:
            return
        self._identify_at = speech_s + 0.25
        emb = self.embedder.embed(self._user_audio())
        scores = self._rows @ emb
        best = int(np.argmax(scores))
        score = float(scores[best])
        self.timeline.append({"t": round(at / SR, 3), "kind": "identify", "score": round(score, 3),
                              "id": self._ids[best], "turn": self.turns})
        by = self._named_by(scores, best)
        if by:
            self._identified = True
            self._ref, self._ref_kind = self._rows[best], "person"
            self._set_speaker(self._ids[best], score, by, events, at)
            if not self._allowed(self._ids[best]):
                self._reject(events, at, score)
        elif speech_s >= self.cfg.reject_after_s:
            if self._strict and self.cfg.reject_late and self._wake_close:
                # A wake word close to someone the device answers: decided at the end of the turn.
                # A first look here, where the session used to be turned away, names a user who
                # has said the command already, so a TV that talks on doesn't hold the turn open.
                last = speech_s >= self.cfg.reject_late_s
                if self._isolated_look(events, at, last=last) or not last:
                    return
            self._identified = True
            self._set_speaker(None, score, None, events, at)
            if self._strict:
                self._reject(events, at, score)

    def _identify_heard(self, events: list[dict[str, Any]], at: int, final: bool) -> None:
        """A strict device whose wake word didn't say who: who is talking,
        from the wake word and everything heard since, whatever the voice
        checks made of it. An allowed person is followed from here on.

        What was heard can include a TV, before or after the user's words, so
        it only ever recognizes someone. Turning a voice away is left to the
        user's own speech, and to the end of the session when nobody's speech
        was ever taken for the user's."""
        heard_s = sum(b - a for a, b in self._heard) / SR
        if self._identified or (not final and heard_s < max(self._heard_at, self.cfg.identify_after_s)):
            return
        self._heard_at = heard_s + 0.25
        heard = [self._audio(a, b) for a, b in self._heard]
        # As for a command: what was said, and the wake word in front only when
        # that is too short. Mixed in sooner, a voice close to someone enrolled
        # can be taken for them.
        if heard_s < self.cfg.identify_after_s:
            heard = [self._anchor_speech, *heard]
        audio = np.concatenate(heard)[-4 * SR:]
        if len(audio) < SR:
            if final and heard_s:
                self._identified = True
                self._set_speaker(None, None, None, events, at)
                self._reject(events, at, None)
            return
        scores = self._rows @ self.embedder.embed(audio)
        best = int(np.argmax(scores))
        score = float(scores[best])
        self.timeline.append({"t": round(at / SR, 3), "kind": "identify", "score": round(score, 3),
                              "id": self._ids[best], "turn": self.turns,
                              **({"wake_word": True} if heard_s < self.cfg.identify_after_s else {})})
        by = self._named_by(scores, best)
        if by:
            self._identified = True
            self._ref, self._ref_kind = self._rows[best], "person"
            self._set_speaker(self._ids[best], score, by, events, at)
            if not self._allowed(self._ids[best]):
                self._reject(events, at, score)
        elif final:
            self._identified = True
            self._set_speaker(None, score, None, events, at)
            self._reject(events, at, score)

    def _maybe_own(self, at: int) -> None:
        """A voice that isn't enrolled: once enough of it is in, follow what the
        user actually said rather than the wake word alone."""
        if self._ref_kind not in ("anchor", "first"):
            return
        if sum(b - a for a, b in self._user_speech) < self.cfg.own_after_s * SR:
            return
        own = self.embedder.embed(self._user_audio(4.0))
        ref = 2 * own + (self._anchor_emb if self._anchor_emb is not None else 0)
        self._ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
        self._ref_kind = "own"
        self.timeline.append({"t": round(at / SR, 3), "kind": "reference", "turn": self.turns})

    def _after_first_turn(self, events: list[dict[str, Any]]) -> None:
        """The first command says more about the user than the wake word did."""
        speech = self._user_audio(6.0)
        with_wake_word = False
        if len(speech) < self.cfg.reject_after_s * SR:
            # A command too short to tell who said it: with the wake word in
            # front it may not be. Only then, since the two together can lift a
            # voice close to someone enrolled over the threshold.
            speech = self._identity_audio(6.0)
            with_wake_word = len(self._anchor_speech) > 0
        if len(speech) < SR:
            if not self._identified and len(self._ids):
                if self._second_look(events) or self._isolated_look(events, self._last_user_end, last=True):
                    return
                if self._strict:
                    self._reject(events, self._last_user_end, None)
            return
        if self._ref_kind not in ("anchor", "first") and (self._identified or not len(self._ids)):
            # Nothing is left to learn from the command: the user is followed
            # by a voiceprint of their own, and either known already or with
            # nobody enrolled to compare them with. Embedding it anyway held
            # the end of every such turn back, by 0.2 s on a GPU for a long one.
            if not self._identified:
                self._identified = True
                self._set_speaker(None, None, None, events, self._last_user_end)
            return
        emb = self.embedder.embed(speech)
        if self._ref_kind in ("anchor", "first"):
            ref = 2 * emb + (self._anchor_emb if self._anchor_emb is not None else 0)
            self._ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
            self._ref_kind = "own"
        if not self._identified:
            self._identified = True
            score = None
            if len(self._ids):
                scores = self._rows @ emb
                best = int(np.argmax(scores))
                score = float(scores[best])
                self.timeline.append({"t": round(self._last_user_end / SR, 3), "kind": "identify",
                                      "score": round(score, 3), "id": self._ids[best], "turn": self.turns,
                                      **({"wake_word": True} if with_wake_word else {})})
                by = self._named_by(scores, best)
                if by:
                    self._ref, self._ref_kind = self._rows[best], "person"
                    self._set_speaker(self._ids[best], score, by, events, self._last_user_end)
                    if not self._allowed(self._ids[best]):
                        self._reject(events, self._last_user_end, score)
                    return
            if len(self._ids) and (self._second_look(events)
                                   or self._isolated_look(events, self._last_user_end, last=True)):
                return
            self._set_speaker(None, score, None, events, self._last_user_end)
            if self._strict and len(self._ids):
                self._reject(events, self._last_user_end, score)

    @property
    def _takes_answers(self) -> bool:
        """Whether this session's first turn can be a short answer
        (``FollowConfig.short_answers``). A kiosk that doesn't say how the
        turn started gets it for a session without a wake word: Home Assistant
        asked a question, it started a conversation or one was started by hand."""
        if self.cfg.short_answers is not None:
            return self.cfg.short_answers
        return self.anchor_audio is None

    def answer_line(self, speech_s: float) -> float:
        """The score an answer of ``speech_s`` seconds needs, with no wake
        word, to count as one of the people the device answers."""
        cfg, m = self.cfg, self.model
        start = m.other_threshold + cfg.answer_line_share * (m.threshold - m.other_threshold)
        share = (speech_s - cfg.short_answer_s) / max(cfg.reject_after_s - cfg.short_answer_s, 1e-9)
        return start + min(1.0, max(0.0, share)) * (m.threshold - start)

    def _answer_audio(self) -> np.ndarray:
        """The audio from the user's first word to their last, widened to
        ``answer_window_s`` and to whole half seconds, as the speaker model
        takes it, with the answer in the middle."""
        a, b = self._user_speech[0][0], self._user_speech[-1][1]
        n = max(int(self.cfg.answer_window_s * SR), -(-(b - a) // QUANTUM) * QUANTUM)
        lo = max(self._base, min((a + b) // 2 - n // 2, self.n - n))
        return self._audio(lo, lo + n)

    def _second_look(self, events: list[dict[str, Any]]) -> bool:
        """Before the first turn's voice goes unrecognized: the same turn
        scored once more with the room around it (``_answer_audio``). Speech
        frames put together are what recognition works on, and for a short
        turn that can fall short where the turn as it was heard does not.

        - At the recognition threshold the person is named, as by any other
          recognition. Of 50 real short commands after a wake word, the
          speech frames recognized 49 and this the one left.
        - Under it, on a device that answers only enrolled or chosen people,
          a reply to a question (``_takes_answers``) is taken when it sounds
          like one of them, by a line that rises with how much was said
          (``answer_line``). Nobody is named then: who is speaking is only
          ever said at the recognition threshold. The closest of their
          voiceprints is the one followed from here on. It is also the one
          Isolation keeps.

        True when the turn was dealt with here, taken or turned away."""
        speech_s = sum(b - a for a, b in self._user_speech) / SR
        if not len(self._ids) or speech_s < self.cfg.min_answer_s:
            return False
        scores = self._rows @ self.embedder.embed(self._answer_audio())
        at = self._last_user_end
        best = int(np.argmax(scores))
        by = self._named_by(scores, best)
        named = by is not None
        allowed = [i for i, pid in enumerate(self._ids) if self._allowed(pid)]
        line = self.answer_line(speech_s) if self._strict and self._takes_answers and allowed else None
        if not named and line is not None:
            best = max(allowed, key=lambda i: float(scores[i]))
        score = float(scores[best])
        self.timeline.append({"t": round(at / SR, 3), "kind": "identify", "score": round(score, 3),
                              "id": self._ids[best], "turn": self.turns, "room": True,
                              **({"answer": True, "line": round(line, 3), "speech_s": round(speech_s, 2)}
                                 if line is not None else {})})
        if named:
            self._identified = True
            self._ref, self._ref_kind = self._rows[best], "person"
            self._set_speaker(self._ids[best], score, by, events, at)
            if not self._allowed(self._ids[best]):
                self._reject(events, at, score)
            return True
        if line is None or score < line:
            return False
        self._identified = True
        self._ref, self._ref_kind = self._rows[best], "person"
        self._set_speaker(None, score, "answer", events, at, line=line)
        return True

    def _isolated_look(self, events: list[dict[str, Any]], at: int, last: bool) -> bool:
        """Before a held first turn is turned away (``reject_late``): the turn
        through Isolation with the voiceprint of the closest person the device
        answers, and the loudest ``answer_window_s`` of what comes back scored
        as any recognition. Next to a TV as loud as the user, what the kiosk
        heard is too much the TV to recognize them by, and what Isolation
        keeps of it is not.

        True when it named that person. Twice a session at most: once
        ``reject_after_s`` of speech is in, and for the ``last`` time when the
        turn ends or ``reject_late_s`` is in. Only with a wake word that came
        close to them."""
        if (not self.cfg.reject_late or not self._strict or self._likely is None or not self._wake_close
                or self._looks >= (2 if last else 1)
                or self.isolator is None or self.model.key != ISOLATOR_MODEL or not self._held):
            return False
        speech = sum(b - a for a, b in self._user_speech)
        if self._looks and speech - self._look_speech < int(self.cfg.min_answer_s * SR):
            return False  # nothing was said since the first look
        self._looks, self._look_speech = (2 if last else 1), speech
        held = list(self._held)
        start, end = held[0][0], held[-1][1]
        if start < self._base:
            return False
        row = self._rows[self._likely]
        ctx = max(self._base, start - int(self.cfg.isolation_context_s * SR))
        t0 = time.perf_counter()
        try:
            around = self.isolator.clean(self._audio(ctx, self.n).copy(), row)
        except IsolationFailed as err:
            _LOG.warning("Isolation not used to recognize this command: %s", err)
            return False
        ms = round((time.perf_counter() - t0) * 1000)
        self.isolated_ms += ms
        went_in = np.concatenate([self._audio(a, b) for a, b in held])
        came_back = np.concatenate([around[a - ctx:b - ctx] for a, b in held])
        kept_db = 20 * np.log10((float(np.sqrt(np.mean(came_back ** 2))) + 1e-9)
                                / (float(np.sqrt(np.mean(went_in ** 2))) + 1e-9))
        step = {"t": round(at / SR, 3), "kind": "identify", "id": self._ids[self._likely], "turn": self.turns,
                "isolated": True, "ms": ms}
        if kept_db < self.cfg.isolation_floor_db:
            # It hears none of them in the turn.
            self.timeline.append({**step, "kept_db": round(kept_db)})
            return False
        la, lb = self._loudest(around, ctx, start, end)
        n = min(len(around), max(int(self.cfg.answer_window_s * SR), -(-(lb - la) // QUANTUM) * QUANTUM))
        lo = max(0, min((la + lb) // 2 - n // 2 - ctx, len(around) - n))
        scores = self._rows @ self.embedder.embed(around[lo:lo + n])
        best = int(np.argmax(scores))
        score = float(scores[best])
        self.timeline.append({**step, "id": self._ids[best], "score": round(score, 3)})
        by = self._named_by(scores, best) if best == self._likely else None
        if not by:
            return False
        self._identified = True
        self._ref, self._ref_kind = row, "person"
        self._set_speaker(self._ids[best], score, by, events, at)
        self._look = (ctx, self.n, row, around)
        return True

    def _named_by(self, scores: np.ndarray, best: int, by: str = "voice") -> str | None:
        """How the closest voiceprint, ``best``, names the speaker: ``by`` at
        the recognition threshold, ``face`` under it when Face Recognition
        says that same person is at the kiosk and the score reaches the face
        line (``face_line_share``), None otherwise. A face only ever confirms
        the closest voice, it never names one on its own."""
        score = float(scores[best])
        if score >= self.model.threshold:
            return by
        at_kiosk = self.cfg.face() if self.cfg.face is not None else None
        if not at_kiosk or self._ids[best] != at_kiosk:
            return None
        m = self.model
        line = m.other_threshold + self.cfg.face_line_share * (m.threshold - m.other_threshold)
        if score < line:
            return None
        self.timeline.append({"t": round(self.n / SR, 3), "kind": "face", "id": at_kiosk, "score": round(score, 3),
                              "line": round(line, 3), "turn": self.turns})
        return "face"

    @property
    def _strict(self) -> bool:
        """Only enrolled voices get an answer."""
        return self.cfg.policy in ("enrolled", "people")

    def _allowed(self, person_id: str) -> bool:
        return self.cfg.policy != "people" or person_id in self.cfg.people

    def _reject(self, events: list[dict[str, Any]], at: int, score: float | None) -> None:
        self._rejected = True
        events.append({"type": "rejected", "t": round(at / SR, 3),
                       "score": None if score is None else round(score, 3)})

    def _set_speaker(self, person_id: str | None, score: float | None, by: str | None,
                     events: list[dict[str, Any]], at: int, line: float | None = None) -> None:
        """``by``: ``wake_word``, ``voice``, ``face`` (a voice under the
        threshold the face at the kiosk confirmed) or ``conversation`` for someone
        recognized, ``answer`` for a short answer taken without a name
        (``line``: the score it needed), None for a voice nobody knows."""
        person = self.people.get(person_id) if person_id else None
        self.speaker = {
            "id": person.id if person else None,
            "name": person.name if person else None,
            "color": person.color if person else None,
            "score": None if score is None else round(score, 3),
            "by": by,
            **({"line": round(line, 3)} if line is not None else {}),
        }
        events.append({"type": "speaker", "t": round(at / SR, 3), **self.speaker})

    # ── output ────────────────────────────────────────────────────────────

    def _merge(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Neighboring audio events become one, so the socket sends fewer
        frames, up to MAX_AUDIO_EVENT samples each."""
        out: list[dict[str, Any]] = []
        for e in events:
            if (e["type"] == "audio" and out and out[-1]["type"] == "audio"
                    and not e.get("keepalive") and not out[-1].get("keepalive")
                    and len(out[-1]["audio"]) + len(e["audio"]) <= MAX_AUDIO_EVENT):
                out[-1] = {**out[-1], "audio": np.concatenate([out[-1]["audio"], e["audio"]])}
            else:
                out.append(e)
        return out

    def raw_audio(self) -> np.ndarray:
        return np.concatenate(self._raw) if self._raw else np.zeros(0, dtype=np.float32)

    def cleaned_audio(self) -> np.ndarray:
        """What the backend got: the user's speech, other voices left out."""
        return np.concatenate(self._sent) if self._sent else np.zeros(0, dtype=np.float32)
