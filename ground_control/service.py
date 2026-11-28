"""The running service: models, people and the active speaker model.

Changing the speaker model rebuilds every voiceprint from the kept audio in
the background. Sessions keep using the old model until the rebuild is done.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from .config import Config
from .engine import Embedder, malloc_trim
from .models import (ALIGNER_FILE, ALIGNER_SHA256, ISOLATOR_CPU_FILE, ISOLATOR_CPU_SHA256, ISOLATOR_FILE,
                     ISOLATOR_MODEL, ISOLATOR_SHA256, SPEAKER_MODELS, VAD_FILE, VAD_SHA256, VISION_FEATURES,
                     VISION_MODELS, VULKAN_ACCURATE_MS, SpeakerModel, apple_silicon, default_vision_keys,
                     verified_path)
from .hub import Hub
from .people import People
from .session import Session, SessionConfig
from .faces import Faces
from .sessions import SessionStore
from .vad import VadModel

_LOG = logging.getLogger(__name__)

HAND_SENSITIVITY = 50
"""Hand sensitivity's default."""


@dataclass
class Status:
    model: str
    rebuilding: str | None = None
    rebuild_done: int = 0
    rebuild_total: int = 0
    sessions: int = 0
    sessions_total: int = 0
    errors: list[str] = field(default_factory=list)


class GroundControl:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.app_path: str | None = None
        """As a Home Assistant app, its page there (``/app/<slug>``, ``/hassio/ingress/<slug>`` before Home Assistant 2026.2)."""
        self.vad = VadModel(verified_path(cfg.models_dir, VAD_FILE, VAD_SHA256))
        self.people = People(cfg.data_dir, self.vad)
        self.faces = Faces(cfg.data_dir)
        self.at_kiosk: dict[str, str | None] = {}
        """Who Face Recognition says is at each kiosk now, by device id."""
        self.sessions = SessionStore(cfg.data_dir, keep_days=cfg.keep_days, keep_audio=cfg.keep_audio)
        self.hub = Hub(cfg.data_dir / "kiosks.json")
        if self.hub.new:
            self.hub.approve_known(self.sessions.list(limit=100000))
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vm")
        self._openvino = None
        """The Intel GPU of the OpenVINO image. Checked here, so an image
        without a usable GPU stops at once with the reason."""
        if cfg.openvino:
            from .ov import Gpu

            self._openvino = Gpu(cfg.openvino_device, cfg.data_dir / "openvino")
            _LOG.info("OpenVINO on %s", self._openvino.name)
        self._vulkan = None
        """The GPU of the Vulkan image (``vulkan.Gpu``). Found here, in a
        process of its own, so an image without a usable GPU stops at once
        with the reason. Each model runs on it when it is faster there than
        on the CPU."""
        self._vulkan_accurate: tuple[str, ...] = ()
        """The features that are Accurate by default for how fast the Vulkan image's GPU runs them."""
        self._vulkan_slow: str | None = None
        """Why Isolation runs on the CPU in the Vulkan image, when the GPU turned out slower."""
        if cfg.vulkan:
            from .vulkan import Gpu as VulkanGpu

            self._vulkan = VulkanGpu(cfg.vulkan_device, cfg.threads)
            _LOG.info("Vulkan on %s", self._vulkan.name)
            self._vulkan_accurate = self._time_accurate()
        self._embedders: dict[str, Embedder] = {}
        self._settings_file = cfg.data_dir / "settings.json"
        saved = self._saved()
        chosen = saved.get("model")
        self.status = Status(model=chosen if chosen in SPEAKER_MODELS else cfg.model)
        self.sessions.keep_days = int(saved.get("keep_days", cfg.keep_days))
        self.sessions.keep_audio = bool(saved.get("keep_audio", cfg.keep_audio))
        self.sessions.keep_frames = bool(saved.get("keep_frames", True))
        self._hand_sensitivity = int(saved.get("hand_sensitivity", HAND_SENSITIVITY))
        self.isolate = bool(saved.get("isolator", False))
        self._isolator = None
        self._isolator_lock = threading.Lock()
        self.align = bool(saved.get("aligner", True))
        self.short_answers = bool(saved.get("short_answers", True))
        """Take a short reply to a question Home Assistant asked when it
        sounds like one of the people the kiosk answers (``FollowConfig.short_answers``)."""
        """Leave the end of the wake word out of the command, for a kiosk
        with "Talk right after the wake word" on."""
        self._aligner = None
        self._aligner_failed = False
        self._aligner_lock = threading.Lock()
        self.vision_keys = self._vision_keys(saved)
        """Vision Match's model for each feature (face, hands, recognition):
        Light or Accurate, as chosen or this machine's default."""
        self._vision = None
        self._vision_error: str | None = None
        self._vision_lock = threading.Lock()
        self.active = self._load(self.status.model)
        # Commands are no longer added to voices on their own. The ones that were are removed once.
        cleared = self.people.clear_learned()
        if cleared:
            _LOG.info("removed the commands that were added to %d voice%s on their own, their voiceprints are built again "
                      "from the enrollments", len(cleared), "" if len(cleared) == 1 else "s")
        if self.people.needs_rebuild(self.status.model):
            _LOG.info("building voiceprints for %s", self.status.model)
            self.people.rebuild(self.status.model, self.active)
        if self.isolate:
            self.isolator()

    def _saved(self) -> dict[str, object]:
        try:
            return json.loads(self._settings_file.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, **values: object) -> None:
        data = {**self._saved(), **values}
        tmp = self._settings_file.with_suffix(".part")
        tmp.write_text(json.dumps(data))
        tmp.replace(self._settings_file)

    @property
    def model(self) -> SpeakerModel:
        """The active model, with any thresholds set on the Settings page."""
        base = SPEAKER_MODELS[self.status.model]
        tuned = (self._saved().get("thresholds") or {}).get(base.key) or {}
        return replace(base, threshold=float(tuned.get("threshold", base.threshold)),
                       other_threshold=float(tuned.get("other_threshold", base.other_threshold)))

    @property
    def gpu(self) -> str:
        """What this install's models reach a GPU through: ``cuda``,
        ``openvino``, ``vulkan`` or nothing when everything runs on the CPU.
        The CUDA image without a usable GPU runs on the CPU. So does the
        Vulkan image on a GPU that is slower than the CPU for the speaker
        model and for the Accurate vision models."""
        if self.cfg.cuda:
            return "cuda" if self.active.on_gpu else ""
        if self._openvino is not None:
            return "openvino"
        if self._vulkan is not None:
            return "vulkan" if self.active.on_gpu or self._vulkan_accurate else ""
        return ""

    def settings(self) -> dict[str, object]:
        m, base = self.model, SPEAKER_MODELS[self.status.model]
        return {"model": self.status.model, "default_model": self.cfg.model, "rebuilding": self.status.rebuilding,
                "models": {k: v.label for k, v in SPEAKER_MODELS.items()},
                "threshold": m.threshold, "other_threshold": m.other_threshold,
                "default_threshold": base.threshold, "default_other_threshold": base.other_threshold,
                "keep_days": self.sessions.keep_days,
                "keep_audio": self.sessions.keep_audio, "keep_frames": self.sessions.keep_frames, "cuda": self.cfg.cuda, "on_gpu": self.active.on_gpu,
                "openvino": self._openvino.name if self._openvino is not None else None,
                "vulkan": self._vulkan.name if self._vulkan is not None else None, "vulkan_slow": self._vulkan_slow,
                "vulkan_on_cpu": self._vulkan.on_cpu() if self._vulkan is not None else [],
                "threads": self.cfg.threads, "port": self.cfg.port,
                "isolator": self.isolate, "isolator_available": self._isolator_files() != [],
                "isolator_model": ISOLATOR_MODEL,
                "isolator_on_gpu": self._isolator.on_gpu if self._isolator is not None else None,
                "aligner": self.align, "aligner_missing": self._aligner_missing(),
                "short_answers": self.short_answers,
                "vision_models": dict(self.vision_keys),
                "vision_defaults": self._vision_defaults(),
                "vision_choices": {f: {k: VISION_MODELS[f][k].label for k in keys}
                                   for f, keys in self.vision_choices().items()},
                "vision_loaded": self._vision is not None,
                "vision_on_gpu": self._vision.on_gpu if self._vision is not None else None,
                # In the Vulkan image each model runs where it is faster: which of the three are on the GPU.
                "vision_gpu_models": ([f for f, m in (("face", self._vision.face), ("hands", self._vision.hand),
                                                      ("recognition", self._vision.recognizer)) if m.on_gpu]
                                      if self._vision is not None else None),
                "vision_max_looks": self._vision.max_looks if self._vision is not None else None,
                "vision_ms": self._vision.ms if self._vision is not None else None,
                "vision_error": self._vision_error,
                "face_threshold": self.face_threshold,
                "hand_sensitivity": self.hand_sensitivity,
                "hand_min_size": round(self.hand_min_size, 3),
                "default_face_threshold": VISION_MODELS["recognition"][self.vision_keys["recognition"]].threshold,
                "faces_enrolled": sum(1 for p in self.people.all() if self.faces.has_face(p.id))}

    def update_settings(self, values: dict[str, object]) -> None:
        """Thresholds (for the active model) and session keeping. Checked, then saved."""
        saved = self._saved()
        thresholds = dict(saved.get("thresholds") or {})
        tuned = dict(thresholds.get(self.status.model) or {})
        for key in ("threshold", "other_threshold"):
            if key in values:
                v = float(values[key])
                if not 0.0 < v < 1.0:
                    raise ValueError(f"{key} must be between 0 and 1")
                tuned[key] = round(v, 3)
        m = SPEAKER_MODELS[self.status.model]
        if tuned.get("other_threshold", m.other_threshold) >= tuned.get("threshold", m.threshold):
            raise ValueError("the other-voice threshold must be below the threshold")
        thresholds[self.status.model] = tuned
        out: dict[str, object] = {"thresholds": thresholds}
        if "keep_days" in values:
            days = int(values["keep_days"])
            if not 1 <= days <= 90:
                raise ValueError("keep_days must be 1 to 90")
            self.sessions.keep_days = out["keep_days"] = days
        if "keep_audio" in values:
            self.sessions.keep_audio = out["keep_audio"] = bool(values["keep_audio"])
        if "keep_frames" in values:
            self.sessions.keep_frames = out["keep_frames"] = bool(values["keep_frames"])
        if "isolator" in values:
            on = bool(values["isolator"])
            if on and not self._isolator_files():
                raise ValueError("the isolation model is not in this image")
            self.isolate = out["isolator"] = on
            if on:
                self.isolator()  # load it now, so the first command doesn't wait
        if "aligner" in values:
            self.align = out["aligner"] = bool(values["aligner"])
        if "short_answers" in values:
            self.short_answers = out["short_answers"] = bool(values["short_answers"])
        if "face_threshold" in values:
            v = float(values["face_threshold"])
            if not 0.0 < v < 1.0:
                raise ValueError("face_threshold must be between 0 and 1")
            out["face_thresholds"] = {**(saved.get("face_thresholds") or {}),
                                      self.vision_keys["recognition"]: round(v, 3)}
        if "hand_sensitivity" in values:
            v = int(values["hand_sensitivity"])
            if not 1 <= v <= 100:
                raise ValueError("hand_sensitivity must be 1 to 100")
            self._hand_sensitivity = out["hand_sensitivity"] = v
        if "vision_model" in values:
            # One key for every feature, as before the models were chosen per feature.
            values = {**values, "vision_models": {f: values["vision_model"] for f in VISION_FEATURES}}
        if "vision_models" in values:
            asked = values["vision_models"]
            if not isinstance(asked, dict) or not set(asked) <= set(VISION_FEATURES):
                raise ValueError("vision_models must map face, hands or recognition to a model")
            choices = self.vision_choices()
            for f, key in asked.items():
                if key not in choices[f]:
                    raise ValueError(f"the {f} model must be one of " + ", ".join(choices[f]))
            keys = {**self.vision_keys, **asked}
            if keys != self.vision_keys:
                self.vision_keys = keys
                out["vision_models"] = dict(keys)
                self._switch_vision()
        self._save(**out)

    def _aligner_missing(self) -> str | None:
        """What finding the end of a wake word needs that this install doesn't
        have: ``model`` or ``espeak-ng``. None when it has both."""
        from .aligner import espeak

        if not (self.cfg.models_dir / ALIGNER_FILE).is_file():
            return "model"
        return None if espeak() else "espeak-ng"

    def aligner(self):
        """The Wake Word Aligner, loaded with the first command that needs
        it. None when it is off or can't run here."""
        if not self.align or self._aligner_failed:
            return None
        with self._aligner_lock:
            if self._aligner is None and not self._aligner_failed:
                from .aligner import Aligner

                missing = self._aligner_missing()
                if missing:
                    self._aligner_failed = True
                    _LOG.warning("the Wake Word Aligner can't run: %s", "its model is not in this install"
                                 if missing == "model" else "espeak-ng is not installed")
                    return None
                try:
                    self._aligner = Aligner(verified_path(self.cfg.models_dir, ALIGNER_FILE, ALIGNER_SHA256),
                                            threads=self.cfg.threads)
                except Exception as err:  # noqa: BLE001 - reported on the status page
                    self._aligner_failed = True
                    self.status.errors.append(f"Wake Word Aligner failed to load: {err}")
                    _LOG.exception("Wake Word Aligner failed to load")
                    return None
                _LOG.info("Wake Word Aligner %s loaded", self._aligner.version)
            return self._aligner

    def vision_choices(self) -> dict[str, list[str]]:
        """For each feature, the models this install has the files of."""
        return {f: [k for k, m in VISION_MODELS[f].items() if (self.cfg.models_dir / m.file).is_file()]
                for f in VISION_FEATURES}

    def _vision_defaults(self) -> dict[str, str]:
        return default_vision_keys(platform.machine(), self.cfg.cuda, apple_silicon(), self._vulkan_accurate)

    def _time_accurate(self) -> tuple[str, ...]:
        """In the Vulkan image: the features whose Accurate model the GPU
        runs fast enough to be the default. Face Recognition isn't timed,
        since it follows the CPU as in the other images."""
        fast = []
        for f in ("face", "hands"):
            part = VISION_MODELS[f]["accurate"]
            if not (self.cfg.models_dir / part.file).is_file():
                continue
            try:
                ms = self._vulkan.time(verified_path(self.cfg.models_dir, part.file, part.sha256))
            except Exception:  # noqa: BLE001 - a default, not worth stopping the service for
                _LOG.exception("couldn't time %s on the GPU, so its default stays Light", part.file)
                continue
            if ms is not None and ms <= VULKAN_ACCURATE_MS:
                fast.append(f)
            _LOG.info("%s by default: %s", {"face": "Face Detection", "hands": "Hand Gestures"}[f],
                      "Light, since the Accurate model runs on the CPU" if ms is None
                      else f"{'Accurate' if f in fast else 'Light'}, since the Accurate model takes {ms:.0f} ms on the GPU")
        return tuple(fast)

    def _vision_keys(self, saved: dict) -> dict[str, str]:
        """Each feature's model: as saved, else as saved before the models
        were chosen per feature (one key for all), else this machine's
        default. A model whose file isn't here falls back to one that is."""
        chosen = saved.get("vision_models")
        if not isinstance(chosen, dict):
            old = saved.get("vision_model")
            chosen = {f: old for f in VISION_FEATURES} if old else {}
        defaults, choices, keys = self._vision_defaults(), self.vision_choices(), {}
        for f in VISION_FEATURES:
            for key in (chosen.get(f), defaults[f], *choices[f]):
                if key in choices[f]:
                    keys[f] = key
                    break
            else:
                keys[f] = defaults[f]
        return keys

    def vision(self):
        """Vision Match's models, loaded with the first look a kiosk asks for.
        None when they can't run here (the reason is in ``settings()``)."""
        with self._vision_lock:
            if self._vision is None and self._vision_error is None:
                from .vision import Vision

                choices = self.vision_choices()
                if any(self.vision_keys[f] not in choices[f] for f in VISION_FEATURES):
                    self._vision_error = "Vision Match's models are not in this install"
                    _LOG.warning("Vision Match can't run: its models are not in this install")
                    return None
                names = ", ".join(f"{f} {VISION_MODELS[f][k].label}" for f, k in self.vision_keys.items())
                try:
                    self._vision = Vision(self.cfg.models_dir, self.vision_keys, threads=self.cfg.threads,
                                          use_cuda=self.cfg.cuda, keep_memory=self.cfg.keep_gpu_memory,
                                          openvino=self._openvino if self._openvino is not None else self._vulkan)
                except Exception as err:  # noqa: BLE001 - reported on the status page
                    self._vision_error = f"Vision Match's models ({names}) failed to load: {err}"
                    self.status.errors.append(self._vision_error)
                    _LOG.exception("Vision Match's models (%s) failed to load", names)
                    return None
                ms = self._vision.ms
                _LOG.info("Vision Match models loaded on %s (%s): face look %s ms, hand look %s ms, "
                          "%s ms a face recognized%s", "the GPU" if self._vision.on_gpu else "the CPU", names,
                          ms["face"], ms["hands"], ms["recognition"],
                          ". They keep their GPU memory between looks" if self._vision.keeps_memory else "")
                people = [p.id for p in self.people.all()]
                rec = self._vision.rec_key
                if self.faces.needs_rebuild(rec, people):
                    t0 = time.perf_counter()
                    self.faces.rebuild(rec, self._vision.recognizer, people)
                    _LOG.info("faceprints rebuilt for the %s recognition model in %.1f s",
                              VISION_MODELS["recognition"][rec].label, time.perf_counter() - t0)
            return self._vision

    @property
    def face_threshold(self) -> float:
        """Face Recognition's threshold for the chosen recognition model, as tuned or its default."""
        key = self.vision_keys["recognition"]
        tuned = (self._saved().get("face_thresholds") or {}).get(key)
        return float(tuned) if tuned is not None else VISION_MODELS["recognition"][key].threshold

    @property
    def hand_sensitivity(self) -> int:
        """Hand sensitivity, 1 to 100: higher counts smaller hands."""
        return self._hand_sensitivity

    @property
    def hand_min_size(self) -> float:
        """The smallest hand counted, as the longer side of its box over the
        frame's: about 9% at the default 50, 13% at 30 and anything at 100.
        Hands that gave a finger count on kiosk frames were 10% or more."""
        return max(0.0, 0.2 * (100 - self.hand_sensitivity) / 100 - 0.01)

    def face_embedders(self) -> dict:
        """The loaded recognition model, so an enrollment lands in its faceprints."""
        vision = self._vision
        return {vision.rec_key: vision.recognizer} if vision is not None else {}

    def _switch_vision(self) -> None:
        """Drop the loaded models. The next look loads the chosen ones."""
        with self._vision_lock:
            old, self._vision, self._vision_error = self._vision, None, None
        if old is not None:
            old.close()
            malloc_trim()
            self.vision()

    def _isolator_files(self) -> list[tuple[str, str, bool]]:
        """The isolation models in this image, in the order to try them: the
        full one on a GPU, then the one made for the CPU."""
        files = [(ISOLATOR_FILE, ISOLATOR_SHA256, True)] if self.cfg.cuda or self._vulkan is not None else []
        files += [(ISOLATOR_CPU_FILE, ISOLATOR_CPU_SHA256, False), (ISOLATOR_FILE, ISOLATOR_SHA256, False)]
        return [f for f in files if (self.cfg.models_dir / f[0]).is_file()]

    def isolator(self):
        """Isolation's model when Isolation is on and the Accurate model is active, else None."""
        if not self.isolate or self.status.model != ISOLATOR_MODEL:
            return None
        with self._isolator_lock:
            if self._isolator is None:
                from .isolator import Isolator

                try:
                    for file, sha, gpu in self._isolator_files():
                        worker = None
                        if gpu and self._vulkan is not None:
                            from .isolator import WINDOW
                            from .vulkan import Slow

                            made_for_cpu = self.cfg.models_dir / ISOLATOR_CPU_FILE
                            try:
                                worker = self._vulkan.isolation(verified_path(self.cfg.models_dir, file, sha), WINDOW,
                                                                made_for_cpu if made_for_cpu.is_file() else None)
                            except Slow as err:
                                self._vulkan_slow = str(err)
                                _LOG.warning("%s", err)
                                continue
                        candidate = Isolator(verified_path(self.cfg.models_dir, file, sha),
                                             threads=self.cfg.threads, use_cuda=gpu and self.cfg.cuda,
                                             keep_memory=self.cfg.keep_gpu_memory, vulkan=worker)
                        if candidate.on_gpu or not gpu:
                            self._isolator = candidate
                            break
                    if self._isolator is None:
                        raise FileNotFoundError("no isolation model in this image")
                except Exception as err:  # noqa: BLE001 - reported on the status page
                    self.isolate = False
                    self.status.errors.append(f"Isolation model failed to load: {err}")
                    _LOG.exception("Isolation model failed to load")
                    return None
                where = (f"{self._vulkan.name} through Vulkan" if self._isolator.vulkan
                         else "the GPU" if self._isolator.on_gpu else "the CPU")
                if not self._isolator.on_gpu:
                    _LOG.warning("Isolation runs on the CPU: each command waits half a second or more for it")
                _LOG.info("Isolation model %s loaded on %s%s", self._isolator.version, where,
                          ", keeping its GPU memory between commands" if self._isolator.keeps_memory else "")
            return self._isolator

    def _load(self, key: str) -> Embedder:
        if key not in self._embedders:
            m = SPEAKER_MODELS[key]
            path = verified_path(self.cfg.models_dir, m.file, m.sha256)
            self._embedders[key] = Embedder(path, threads=self.cfg.threads, use_cuda=self.cfg.cuda,
                                            remote=self._vulkan.model(path) if self._vulkan is not None else None)
            where = "the GPU" if self._embedders[key].on_gpu else "the CPU"
            _LOG.info("%s model loaded on %s", m.label, where)
        return self._embedders[key]

    def new_session(self, **options: object) -> Session:
        cfg = SessionConfig(**{k: v for k, v in options.items() if v is not None})
        return Session(cfg, self.model, self.active, self.vad, self.people)

    def new_follow(self, policy: str = "anyone", anchor=None, people: tuple[str, ...] = (),
                   speaker: str | None = None, isolation: bool = True,
                   wake: dict[str, object] | None = None, asked: bool | None = None,
                   face: Callable[[], str | None] | None = None, align: bool = True) -> "Follow":
        """``isolation``: the kiosk's own switch. Isolation runs when that and
        the service's setting are both on.

        ``wake``: what the kiosk said about the wake word (``wake_details``).
        The end of the wake word is looked for only when the kiosk has "Talk
        right after the wake word" on for this session, which is when it
        reaches the assistant. A follow-up has no wake word of its own.

        ``asked``: the turn is a reply to something Home Assistant asked or a
        follow-up, as the kiosk says. None when it doesn't say.

        ``face``: who is at the kiosk, for a kiosk with Confirm voices with
        faces on (``FollowConfig.face``).

        ``align``: the kiosk's own Wake Word Aligner switch. The end of the
        wake word is looked for when that and the service's setting are both on."""
        from .follow import Follow, FollowConfig

        find = None
        phrase = (wake or {}).get("wake_word")
        if align and wake and wake.get("seamless") and phrase and anchor is not None and speaker is None:
            aligner = self.aligner()
            if aligner is not None:
                language = wake.get("language")

                def find(anchor_audio, head):
                    return aligner.find(anchor_audio, head, str(phrase), language if isinstance(language, str) else None)

        cfg = FollowConfig(policy=policy, people=people, short_answers=asked if self.short_answers else False,
                           face=face)
        return Follow(cfg, self.model, self.active, self.vad, self.people,
                      anchor=anchor, speaker=speaker, isolator=self.isolator() if isolation else None,
                      wake_word_end=find)

    async def run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *args)

    def enroll_models(self) -> dict[str, Embedder]:
        """Every loaded model, so an enrollment lands in each one's voiceprints."""
        return dict(self._embedders)

    async def switch_model(self, key: str) -> None:
        if key not in SPEAKER_MODELS:
            raise ValueError(f"unknown model {key}")
        if key == self.status.model or self.status.rebuilding:
            return
        self.status.rebuilding = key
        try:
            embedder = await self.run(self._load, key)
            await self.run(self.people.rebuild, key, embedder)
            old = self.status.model
            self.active = embedder
            self.status.model = key
            self._save(model=key)
            dropped = self._embedders.pop(old, None)
            if dropped is not None and dropped is not embedder:
                dropped.close()
            malloc_trim()
            _LOG.info("speaker model is now %s", key)
        except Exception as err:  # noqa: BLE001 - reported on the status page
            self.status.errors.append(f"model switch to {key} failed: {err}")
            _LOG.exception("model switch failed")
        finally:
            self.status.rebuilding = None

    def identify(self, audio: np.ndarray) -> dict[str, object]:
        """Who is speaking in a clip, outside a session."""
        from .people import speech_only

        speech = speech_only(self.vad, audio)
        ids, rows = self.people.voiceprints(self.model.key)
        if len(speech) < 16000 or not ids:
            return {"id": None, "name": None, "score": None, "by": None}
        scores = rows @ self.active.embed(speech[-64000:])
        i = int(np.argmax(scores))
        if scores[i] < self.model.threshold:
            return {"id": None, "name": None, "score": round(float(scores[i]), 3), "by": None}
        person = self.people.get(ids[i])
        return {"id": person.id, "name": person.name, "color": person.color,
                "score": round(float(scores[i]), 3), "by": "voice"}

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        if self._vision is not None:
            self._vision.close()
        if self._isolator is not None:
            self._isolator.close()
        # After any write that is under way, so sessions.db is whole on its own.
        self.sessions.close()
