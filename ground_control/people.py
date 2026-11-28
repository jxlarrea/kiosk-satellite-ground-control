"""People, their voices and what is theirs.

On disk, under the data directory:

- ``people.json``: each person's id, name, ring color and things (to-do list,
  calendar, playlist, Home Assistant person).
- ``voices/<id>/enroll/*.wav``: the audio every voiceprint is built from. It
  is kept so a model change rebuilds the voiceprints without anyone recording
  again. Only audio someone chose to add is in it: an enrollment at a kiosk
  or a session's speech added by hand. Up to 2026.11.17 confident commands
  were also added on their own (``learned``), see ``clear_learned``.
- ``profiles/<model>.json``: per person, the sum of unit-length window
  embeddings and their count. The voiceprint is that sum, normalized, so new
  speech folds in as an exact weighted average (Voice Incubator's profile).
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE
from .audio import read_wav, write_wav
from .engine import Embedder
from .vad import VadModel, speech_mask

WINDOW_S = 4.0
HOP_S = 2.0
MIN_SPEECH_S = 1.0
THINGS = ("todo_list", "calendar", "playlist", "person")
COLORS = ("#56814F", "#7A5BA6", "#8A6516", "#3D64A8", "#3F7F86", "#9A4A86")
"""Ring colors that stay distinct on the Echo Dot ring."""

_NAME = re.compile(r"^.{1,40}$")


@dataclass
class Person:
    id: str
    name: str
    color: str
    things: dict[str, str] = field(default_factory=dict)
    created: float = field(default_factory=time.time)


def speech_only(vad: VadModel, audio: np.ndarray) -> np.ndarray:
    return audio[speech_mask(vad, audio)]


def window_embeddings(embedder: Embedder, speech: np.ndarray) -> list[np.ndarray]:
    """4 s windows at a 2 s hop over speech-only audio, or one window if shorter."""
    win, hop = int(WINDOW_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
    if len(speech) < MIN_SPEECH_S * SAMPLE_RATE:
        return []
    if len(speech) <= win:
        return [embedder.embed(speech)]
    return [embedder.embed(speech[s:s + win]) for s in range(0, len(speech) - win + 1, hop)]


class People:
    def __init__(self, data_dir: Path, vad: VadModel) -> None:
        self.dir = data_dir
        self.vad = vad
        self._lock = threading.RLock()
        self._people: dict[str, Person] = {}
        self._profiles: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
        self.on_change: list[Callable[[], None]] = []
        """Called after any change to who is enrolled or what is theirs (from any thread)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── people ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        f = self.dir / "people.json"
        if f.is_file():
            for p in json.loads(f.read_text()).get("people", []):
                self._people[p["id"]] = Person(**p)

    def _save(self) -> None:
        f = self.dir / "people.json"
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps({"people": [asdict(p) for p in self._people.values()]}, indent=1))
        tmp.replace(f)
        self._changed()

    def _changed(self) -> None:
        for callback in self.on_change:
            callback()

    def summary(self) -> list[dict[str, object]]:
        """What kiosks get: enough to show the speaker and resolve "my"."""
        return [{"id": p.id, "name": p.name, "color": p.color, "things": p.things,
                 "has_voice": self.has_voice(p.id)} for p in self.all()]

    def all(self) -> list[Person]:
        with self._lock:
            return sorted(self._people.values(), key=lambda p: p.created)

    def get(self, person_id: str) -> Person | None:
        with self._lock:
            return self._people.get(person_id)

    def add(self, name: str, color: str | None = None) -> Person:
        name = name.strip()
        if not _NAME.match(name):
            raise ValueError("name must be 1 to 40 characters")
        with self._lock:
            if any(p.name.lower() == name.lower() for p in self._people.values()):
                raise ValueError(f"{name} is already enrolled")
            used = {p.color for p in self._people.values()}
            color = color or next((c for c in COLORS if c not in used), COLORS[len(self._people) % len(COLORS)])
            person = Person(id=uuid.uuid4().hex[:12], name=name, color=color)
            self._people[person.id] = person
            self._save()
            return person

    def update(self, person_id: str, *, name: str | None = None, color: str | None = None,
               things: dict[str, str | None] | None = None) -> Person:
        with self._lock:
            person = self._people[person_id]
            if name is not None:
                name = name.strip()
                if not _NAME.match(name):
                    raise ValueError("name must be 1 to 40 characters")
                person.name = name
            if color is not None:
                person.color = color
            for key, value in (things or {}).items():
                if key not in THINGS:
                    raise ValueError(f"unknown thing {key}")
                if value:
                    person.things[key] = value
                else:
                    person.things.pop(key, None)
            self._save()
            return person

    def forget(self, person_id: str) -> None:
        with self._lock:
            self._people.pop(person_id)
            for model in list(self._profiles):
                self._profiles[model].pop(person_id, None)
                self._save_profiles(model)
            self._save()
        voices = self.dir / "voices" / person_id
        if voices.is_dir():
            for f in sorted(voices.rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
            voices.rmdir()

    # ── voices ────────────────────────────────────────────────────────────

    def clips(self, person_id: str) -> list[Path]:
        return sorted((self.dir / "voices" / person_id / "enroll").glob("*.wav"))

    def clear_learned(self) -> list[str]:
        """Delete the commands that were added to voices on their own, and
        the voiceprints they are part of. Returns whose voices had any.

        Up to 2026.11.17 a command recognized with confidence was folded
        into the speaker's voiceprint. Whatever played in the room under it
        went in with it, and a voiceprint made from commands heard with a
        video playing then took the video's voice for the user's. A
        voiceprint is now only made from audio someone chose to add.

        The voiceprints of the people it affected are dropped for every
        model, so they are built again from the enrollments alone."""
        cleared: list[str] = []
        for folder in sorted((self.dir / "voices").glob("*/learned")):
            had = False
            for f in folder.glob("*"):
                had = had or f.suffix == ".wav"
                f.unlink()
            folder.rmdir()
            if had:
                cleared.append(folder.parent.name)
        if cleared:
            with self._lock:
                for f in sorted((self.dir / "profiles").glob("*.json")):
                    profiles = self._model_profiles(f.stem)
                    if any(profiles.pop(pid, None) is not None for pid in cleared):
                        self._save_profiles(f.stem)
        return cleared

    def speech_seconds(self, person_id: str) -> float:
        return sum((f.stat().st_size - 44) / 2 / SAMPLE_RATE for f in self.clips(person_id))

    def has_voice(self, person_id: str) -> bool:
        return bool(self.clips(person_id))

    def enroll(self, person_id: str, audio: np.ndarray, models: dict[str, Embedder],
               source: dict[str, str | None] | None = None) -> float:
        """Add speech to a person. Returns the seconds of speech kept.

        ``source`` says where it was recorded: ``device_id``, ``device_name``
        and ``model`` (the kind of device). A voice is recognized best on the
        kinds of device it was enrolled on, so the People page shows them.
        """
        speech = speech_only(self.vad, audio)
        if len(speech) < MIN_SPEECH_S * SAMPLE_RATE:
            raise ValueError("not enough speech in the recording")
        if person_id not in self._people:
            raise KeyError(person_id)
        eid = str(time.time_ns())
        write_wav(self.dir / "voices" / person_id / "enroll" / f"{eid}.wav", speech)
        record = {"id": eid, "seconds": round(len(speech) / SAMPLE_RATE, 1), "at": time.time(),
                  **{k: (source or {}).get(k) for k in ("device_id", "device_name", "model")}}
        with self._lock:
            records = self._records(person_id)
            records.append(record)
            self._write_records(person_id, records)
        for key, embedder in models.items():
            self._fold(key, person_id, window_embeddings(embedder, speech))
        self._changed()
        return len(speech) / SAMPLE_RATE

    def _records_file(self, person_id: str) -> Path:
        return self.dir / "voices" / person_id / "enrollments.json"

    def _records(self, person_id: str) -> list[dict[str, object]]:
        try:
            return json.loads(self._records_file(person_id).read_text())
        except (OSError, ValueError):
            return []

    def _write_records(self, person_id: str, records: list[dict[str, object]]) -> None:
        f = self._records_file(person_id)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps(records))
        tmp.replace(f)

    def enrollments(self, person_id: str) -> list[dict[str, object]]:
        """Each enrollment, newest first, with where it was recorded when known."""
        known = {r["id"]: r for r in self._records(person_id)}
        out = []
        for clip in self.clips(person_id):
            r = known.get(clip.stem) or {"id": clip.stem, "seconds": round((clip.stat().st_size - 44) / 2 / SAMPLE_RATE, 1),
                                         "at": clip.stat().st_mtime, "device_id": None, "device_name": None, "model": None}
            out.append(r)
        return sorted(out, key=lambda r: r["at"], reverse=True)

    def delete_enrollment(self, person_id: str, enrollment_id: str, models: dict[str, Embedder]) -> None:
        """Remove one enrollment and rebuild the person's voiceprints without it."""
        if not enrollment_id.isdigit():
            raise KeyError(enrollment_id)
        clip = self.dir / "voices" / person_id / "enroll" / f"{enrollment_id}.wav"
        if not clip.is_file():
            raise KeyError(enrollment_id)
        clip.unlink()
        with self._lock:
            self._write_records(person_id, [r for r in self._records(person_id) if r["id"] != enrollment_id])
        for key, embedder in models.items():
            self.rebuild_person(person_id, key, embedder)
        self._changed()

    def rebuild_person(self, person_id: str, model_key: str, embedder: Embedder) -> None:
        embs: list[np.ndarray] = []
        for clip in self.clips(person_id):
            embs.extend(window_embeddings(embedder, read_wav(clip)))
        with self._lock:
            profiles = self._model_profiles(model_key)
            if embs:
                profiles[person_id] = (np.sum(embs, axis=0), len(embs))
            else:
                profiles.pop(person_id, None)
            self._save_profiles(model_key)

    def rebuild(self, model_key: str, embedder: Embedder) -> None:
        """Rebuild every voiceprint for a model from the kept audio."""
        fresh: dict[str, tuple[np.ndarray, int]] = {}
        for person in self.all():
            embs: list[np.ndarray] = []
            for clip in self.clips(person.id):
                embs.extend(window_embeddings(embedder, read_wav(clip)))
            if embs:
                fresh[person.id] = (np.sum(embs, axis=0), len(embs))
        with self._lock:
            self._profiles[model_key] = fresh
            self._save_profiles(model_key)

    # ── voiceprints ───────────────────────────────────────────────────────

    def _profile_file(self, model_key: str) -> Path:
        return self.dir / "profiles" / f"{model_key}.json"

    def _model_profiles(self, model_key: str) -> dict[str, tuple[np.ndarray, int]]:
        if model_key not in self._profiles:
            f = self._profile_file(model_key)
            loaded: dict[str, tuple[np.ndarray, int]] = {}
            if f.is_file():
                for pid, entry in json.loads(f.read_text()).items():
                    loaded[pid] = (np.array(entry["sum"], dtype=np.float32), int(entry["count"]))
            self._profiles[model_key] = loaded
        return self._profiles[model_key]

    def _save_profiles(self, model_key: str) -> None:
        f = self._profile_file(model_key)
        f.parent.mkdir(parents=True, exist_ok=True)
        data = {pid: {"sum": s.tolist(), "count": n} for pid, (s, n) in self._profiles.get(model_key, {}).items()}
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps(data))
        tmp.replace(f)

    def _fold(self, model_key: str, person_id: str, embeddings: list[np.ndarray]) -> None:
        if not embeddings:
            return
        with self._lock:
            profiles = self._model_profiles(model_key)
            total, count = profiles.get(person_id, (np.zeros_like(embeddings[0]), 0))
            profiles[person_id] = (total + np.sum(embeddings, axis=0), count + len(embeddings))
            self._save_profiles(model_key)

    def needs_rebuild(self, model_key: str) -> bool:
        """True when someone has audio but no voiceprint for this model."""
        with self._lock:
            profiles = self._model_profiles(model_key)
            return any(p.id not in profiles and self.has_voice(p.id) for p in self.all())

    def voiceprints(self, model_key: str) -> tuple[list[str], np.ndarray]:
        """Person ids and their unit-length voiceprints, one row each."""
        with self._lock:
            profiles = self._model_profiles(model_key)
            ids = [p.id for p in self.all() if p.id in profiles]
            if not ids:
                return [], np.zeros((0, 0), dtype=np.float32)
            rows = np.stack([profiles[i][0] for i in ids])
        rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)
        return ids, rows
