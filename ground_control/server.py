"""HTTP and WebSocket API: the web UI's, and Kiosk Satellite's sockets.

Everything but ``/health`` and ``/auth`` needs a Home Assistant token. As a
Home Assistant app the panel needs none, since ingress has signed the user in.
With ``HA_TOKEN`` set and taken by Home Assistant, the web UI needs none
either. The kiosk sockets still need the kiosk's own token.
The kiosk sockets are for Kiosk Satellite only (ground_control.clients).
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from . import __version__
from .logs import LOGS, line as log_line
from .audio import float_to_pcm16, pcm16_to_float, read_wav
from .auth import TokenCheck, bearer
from .clients import ClientCheck, challenge
from .enroll import GuidedEnrollment
from .install import describe as describe_install
from .models import SPEAKER_MODELS
from .login import HomeAssistantLogin, LoginError
from .pictures import Pictures
from .suggestions import Suggestions
from .service import GroundControl
from .sessions import new_id
from .updates import UpdateCheck
from .faces import Recognizer
from .vision_sessions import VisionTracker

_LOG = logging.getLogger(__name__)

VM = web.AppKey("vm", GroundControl)
AUTH = web.AppKey("auth", TokenCheck)
OPEN = web.AppKey("open", bool)
PICTURES = web.AppKey("pictures", Pictures)
SUGGESTIONS = web.AppKey("suggestions", Suggestions)
LOGIN = web.AppKey("login", HomeAssistantLogin)
UPDATES = web.AppKey("updates", UpdateCheck)
CLIENTS = web.AppKey("clients", ClientCheck)
SOCKETS = ("/api/session", "/api/enroll", "/api/kiosk", "/api/vision", "/api/face_enroll", "/api/events")
MIN_PARTIAL_ENROLL_S = 10.0
WEB = Path(__file__).resolve().parent / "web"
INGRESS_PROXY = "172.30.32.2"
"""The Supervisor's address. Home Assistant has signed the user in before it proxies."""


def _from_ingress(request: web.Request) -> bool:
    return request.remote == INGRESS_PROXY and "X-Ingress-Path" in request.headers


_REFUSED: dict[tuple[str, ...], float] = {}


def _once_a_minute(*key: str) -> bool:
    """Whether to log a refusal again, so a kiosk retrying doesn't flood the log."""
    if time.monotonic() - _REFUSED.get(key, -60.0) < 60.0:
        return False
    _REFUSED[key] = time.monotonic()
    return True


async def _web_token(request: web.Request) -> bool | None:
    """Whether Home Assistant takes ``HA_TOKEN``: None when it isn't set or couldn't be asked."""
    token = request.app[VM].cfg.ha_token
    return await request.app[AUTH].check(token) if token else None


def _ha_token(request: web.Request) -> str | None:
    """The token to ask Home Assistant with for the web UI: its own, or ``HA_TOKEN``."""
    return bearer(request.headers.get("Authorization")) or request.app[VM].cfg.ha_token


async def _signed_in(request: web.Request, token: str | None = None, web_ui: bool = True) -> bool | None:
    """Whether the request is signed in. None when Home Assistant couldn't be asked, so the
    web UI waits for it instead of asking to log in again."""
    if request.app[OPEN] or _from_ingress(request):
        return True
    web_token = await _web_token(request) if web_ui else None
    if web_token is True:
        return True
    token = token or bearer(request.headers.get("Authorization"))
    ok = await request.app[AUTH].check(token)
    if ok is False and web_ui and web_token is None and request.app[VM].cfg.ha_token:
        ok = None  # HA_TOKEN not checked yet and no token of its own
    if ok is False and _once_a_minute(request.remote or "?", request.path):
        _LOG.warning("sign-in refused for %s from %s: %s", request.path, request.remote or "?",
                     "no token" if not token else "Home Assistant did not accept the token")
    return ok


@web.middleware
async def require_token(request: web.Request, handler):
    public = request.path in ("/", "/health") or request.path.startswith(("/static/", "/auth/"))
    if public or request.path in SOCKETS:
        return await handler(request)
    ok = await _signed_in(request)
    if ok:
        return await handler(request)
    if ok is None:
        raise web.HTTPServiceUnavailable(text=f"Home Assistant at {request.app[AUTH].ha_url} can't be reached")
    raise web.HTTPUnauthorized(text="sign in with a Home Assistant token")


async def _open_socket(request: web.Request, first_type: str, nonce: str | None = None,
                       web_ui: bool = False) -> tuple[web.WebSocketResponse, dict[str, Any] | None]:
    """Accept a socket and check its first message and token, after sending ``nonce`` to sign.
    A kiosk's socket always needs its token, ``HA_TOKEN`` only signs in the web UI's."""
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20)
    await ws.prepare(request)
    if nonce is not None:
        await ws.send_json({"type": "challenge", "nonce": nonce})
    try:
        first = await ws.receive(timeout=5)
    except asyncio.TimeoutError:
        first = None
    if first is not None and first.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
        return ws, None  # the other side left before saying anything
    try:
        start = json.loads(first.data) if first is not None and first.type == WSMsgType.TEXT else {}
    except ValueError:
        start = {}
    if not isinstance(start, dict) or start.get("type") != first_type:
        why = (f"no {first_type} within 5 seconds" if first is None
               else f"the first message was not a {first_type}")
        await _refuse(request, ws, f"no_{first_type}", why, close_code=4400)
        return ws, None
    ok = await _signed_in(request, start.get("token"), web_ui=web_ui)
    if ok is None and web_ui:
        await ws.send_json({"type": "error", "code": "home_assistant_unreachable"})
        await ws.close(code=4503, message=b"home_assistant_unreachable")
        return ws, None
    if not ok:
        await ws.send_json({"type": "error", "code": "unauthorized"})
        await ws.close(code=4401, message=b"unauthorized")
        return ws, None
    return ws, start


async def _refuse(request: web.Request, ws: web.WebSocketResponse, code: str, why: str,
                  close_code: int = 4403) -> None:
    """Close a socket with ``code`` for the kiosk and ``why`` in the log, at most once a minute."""
    if _once_a_minute(request.remote or "?", request.path, code):
        _LOG.warning("%s refused from %s: %s", request.path, request.remote or "?", why)
    await ws.send_json({"type": "error", "code": code})
    await ws.close(code=close_code, message=code.encode())


async def _kiosk_start(request: web.Request) -> tuple[web.WebSocketResponse, dict[str, Any] | None]:
    """A session or enrollment socket: its start must carry the ticket of an approved kiosk."""
    ws, start = await _open_socket(request, "start")
    if start is None:
        return ws, None
    vm = request.app[VM]
    device_id = start.get("device_id")
    if not vm.hub.ticket_ok(device_id if isinstance(device_id, str) else None, start.get("ticket")):
        await _refuse(request, ws, "not_approved",
                      f"{device_id or 'no device_id'} has no ticket from an approved kiosk socket")
        return ws, None
    return ws, start


def _features(value: Any) -> dict[str, Any] | None:
    """The Voice Match and Vision Match settings a kiosk says it has on, kept to what the Kiosks page shows."""
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {k: value[k] is True for k in
                           ("voice_match", "trimming", "isolation", "wake_word_aligner", "speaker_sensor", "show_speaker",
                            "vision_match", "face_detection", "hand_gestures", "face_recognition",
                            "dismiss_on_face", "postpone_on_face", "face_confirms_voice") if k in value}
    gestures = value.get("finger_gestures")
    if isinstance(gestures, int) and not isinstance(gestures, bool) and 0 <= gestures <= 100:
        out["finger_gestures"] = gestures
    if value.get("policy") in ("anyone", "enrolled", "people"):
        out["policy"] = value["policy"]
    people = value.get("people")
    if isinstance(people, list):
        out["people"] = [p for p in people if isinstance(p, str)][:50]
    return out


def _person(vm: GroundControl, p) -> dict[str, Any]:
    enrollments = vm.people.enrollments(p.id)
    faces = vm.faces.enrollments(p.id)
    for f in faces:
        f["crop_names"] = [c.name for c in vm.faces.crops(p.id, f["id"])]
    return {**asdict(p), "speech_s": round(vm.people.speech_seconds(p.id), 1),
            "clips": len(vm.people.clips(p.id)), "has_voice": vm.people.has_voice(p.id),
            "enrollments": enrollments, "models": sorted({e["model"] for e in enrollments if e.get("model")}),
            "picture": bool((p.things or {}).get("person")),
            "has_face": bool(faces), "face_enrollments": faces}


async def person_picture(request: web.Request) -> web.Response:
    """The picture of the Home Assistant person a person is linked to."""
    vm = request.app[VM]
    person = vm.people.get(request.match_info["id"])
    entity = (person.things or {}).get("person") if person else None
    found = await request.app[PICTURES].get(entity or "", _ha_token(request))
    if found is None:
        raise web.HTTPNotFound(text="no picture")
    body, ctype = found
    return web.Response(body=body, content_type=ctype, headers={"Cache-Control": "private, max-age=600"})


async def suggestions(request: web.Request) -> web.Response:
    """What a person's things can be, from Home Assistant, for the People page."""
    return web.json_response(await request.app[SUGGESTIONS].get(_ha_token(request)))


def _source(vm: GroundControl, device_id: str | None, name: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Where an enrollment was recorded, from the kiosk when it is known."""
    k = vm.hub.kiosks.get(device_id or "")
    return {"device_id": device_id, "device_name": (k.name if k else None) or name,
            "model": (k.model if k else None) or model}


def _known(vm: GroundControl, request: web.Request) -> str:
    pid = request.match_info["id"]
    if vm.people.get(pid) is None:
        raise web.HTTPNotFound(text="no such person")
    return pid


async def _json(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as err:
        raise web.HTTPBadRequest(text="send JSON") from err
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="send a JSON object")
    return body


# ── status ────────────────────────────────────────────────────────────────

# ── log in with Home Assistant ──────────────────────────────────────────────

def _origin(request: web.Request) -> str:
    """The address the browser opened Ground Control at, behind a reverse proxy too."""
    proto = request.headers.get("X-Forwarded-Proto", request.scheme).split(",")[0].strip()
    host = request.headers.get("X-Forwarded-Host", request.host).split(",")[0].strip()
    return f"{proto}://{host}"


def _page(title: str, body: str) -> web.Response:
    return web.Response(content_type="text/html", headers={"Cache-Control": "no-store"}, text=(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title>"
        '<body style="font-family:system-ui,sans-serif;margin:48px 24px;max-width:520px">'
        f"{body}</body></html>"))


async def auth_info(request: web.Request) -> web.Response:
    login = request.app[LOGIN]
    token = None
    if request.app[VM].cfg.ha_token and not request.app[OPEN]:
        token = {True: "good", False: "refused", None: "unreachable"}[await _web_token(request)]
    return web.json_response({"login": login.available and not request.app[OPEN],
                              "ha_url": login.ha_url if login.available else None, "token": token})


async def auth_login(request: web.Request) -> web.Response:
    login = request.app[LOGIN]
    if not login.available:
        raise web.HTTPNotFound(text="open Ground Control from the Home Assistant sidebar")
    raise web.HTTPFound(login.start(_origin(request), request.query.get("next", "")))


REFRESH_COOKIE = "gc_refresh"
CLIENT_COOKIE = "gc_client"
COOKIE_DAYS = 90
"""As long as Home Assistant keeps a refresh token that isn't used. Each refresh starts it again."""


def _keep_login(request: web.Request, response: web.StreamResponse, refresh_token: str, client_id: str) -> None:
    """The login in cookies only Ground Control reads: Safari wipes what a page script stores
    after 7 days without a visit, but not these."""
    secure = _origin(request).startswith("https:")
    for name, value in ((REFRESH_COOKIE, refresh_token), (CLIENT_COOKIE, client_id)):
        response.set_cookie(name, value, max_age=COOKIE_DAYS * 86400, path="/auth/", httponly=True,
                            samesite="Lax", secure=secure)


def _drop_login(response: web.StreamResponse) -> None:
    for name in (REFRESH_COOKIE, CLIENT_COOKIE):
        response.del_cookie(name, path="/auth/")


async def auth_callback(request: web.Request) -> web.Response:
    login = request.app[LOGIN]
    if "error" in request.query or "code" not in request.query:
        return _page("Not signed in", "<h1>Not signed in</h1><p>Home Assistant didn't sign you in.</p>"
                     '<p><a href="../">Try again</a></p>')
    try:
        tokens, back = await login.finish(request.query.get("state", ""), request.query["code"])
    except LoginError as err:
        _LOG.warning("login with Home Assistant failed: %s", err)
        return _page("Not signed in", f"<h1>Not signed in</h1><p>{html.escape(str(err))}</p>"
                     '<p><a href="../">Try again</a></p>')
    _LOG.info("signed in with Home Assistant from %s", request.remote)
    # Back to where the browser was. The web UI gets its access token from /auth/refresh.
    response = web.Response(status=302, headers={"Location": "../" + back, "Cache-Control": "no-store"})
    _keep_login(request, response, tokens["refresh_token"], tokens["client_id"])
    return response


async def auth_refresh(request: web.Request) -> web.Response:
    """A new access token from the login cookie. A browser that logged in before the cookie
    sends its refresh token once and gets the cookie for it."""
    token, client_id = request.cookies.get(REFRESH_COOKIE), request.cookies.get(CLIENT_COOKIE)
    if not token or not client_id:
        body = await _json(request) if request.can_read_body else {}
        token, client_id = body.get("refresh_token"), body.get("client_id")
    if not isinstance(token, str) or not isinstance(client_id, str) or not token or not client_id:
        raise web.HTTPUnauthorized(text="not logged in")
    try:
        response = web.json_response(await request.app[LOGIN].refresh(token, client_id),
                                     headers={"Cache-Control": "no-store"})
    except LoginError as err:
        if not err.refused:
            raise web.HTTPBadGateway(text=str(err)) from err
        response = web.Response(status=401, text=str(err))
        _drop_login(response)
        return response
    _keep_login(request, response, token, client_id)
    return response


async def auth_logout(request: web.Request) -> web.Response:
    body = await _json(request) if request.can_read_body else {}
    if isinstance(body.get("access_token"), str):
        request.app[AUTH].forget(body["access_token"])
    for token in {request.cookies.get(REFRESH_COOKIE), body.get("refresh_token")}:
        if isinstance(token, str) and token:
            await request.app[LOGIN].revoke(token)
    _LOG.info("signed out from %s", request.remote)
    response = web.json_response({"signed_out": True})
    _drop_login(response)
    return response


async def health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "version": __version__})


async def status(request: web.Request) -> web.Response:
    vm = request.app[VM]
    return web.json_response({**asdict(vm.status), "version": __version__, "people": len(vm.people.all()),
                              "model_label": vm.model.label, "threshold": vm.model.threshold,
                              "other_threshold": vm.model.other_threshold, "cuda": vm.cfg.cuda,
                              "on_gpu": vm.active.on_gpu, "gpu": vm.gpu, "threads": vm.cfg.threads,
                              "kiosks": sum(k.connected for k in vm.hub.kiosks.values()),
                              "isolation": vm.isolator() is not None,
                              "aligner": vm.align and vm._aligner_missing() is None,
                              "isolation_on_gpu": bool(vm.isolator() and vm.isolator().on_gpu),
                              "update": request.app[UPDATES].available() if UPDATES in request.app else None})

MISSING_PEOPLE_FIX = "choose them again in Kiosk Satellite, under Voice Satellite > Ground Control Voice Match > People"


def _check_people(vm: GroundControl, device_id: str | None, people: Any) -> list[str]:
    """The people a kiosk answers that Ground Control doesn't have, as after
    moving to another Ground Control. Logged once for each set, since
    otherwise every command is rejected without a word why."""
    ids = [p for p in people if isinstance(p, str)] if isinstance(people, (list, tuple)) else []
    missing = [p for p in ids if vm.people.get(p) is None]
    k = vm.hub.kiosks.get(device_id or "")
    if k is not None and tuple(missing) != k._missing_warned:
        k._missing_warned = tuple(missing)
        if missing:
            n = f"{len(missing)} {'person' if len(missing) == 1 else 'people'}"
            known = len(ids) - len(missing)
            _LOG.warning("kiosk %s answers %s Ground Control doesn't have%s: %s", k.name, n,
                         "" if known else ", so it answers nobody", MISSING_PEOPLE_FIX)
    return missing


def _check_features(vm: GroundControl, device_id: str | None, features: dict[str, Any] | None) -> None:
    if features and features.get("policy") == "people":
        _check_people(vm, device_id, features.get("people"))


# ── sessions ──────────────────────────────────────────────────────────────

WAKE_TEXT = ("wake_word", "wake_word_engine", "wake_word_model", "language")
INTERACTIONS = ("wake_word", "question", "ask_question", "start_conversation", "follow_up", "manual")
ASKED = ("question", "ask_question", "start_conversation", "follow_up")
"""The turns that are a reply to something Home Assistant said."""


def wake_details(start: dict[str, Any]) -> dict[str, Any]:
    """What a kiosk says about how a session started, all of it optional (an
    older Kiosk Satellite sends none):

    - ``wake_word``: the wake word as it is said, such as "Okay Nabu".
    - ``wake_word_engine`` and ``wake_word_model``: what heard it.
    - ``seamless``: "Talk right after the wake word" is on, so the stream
      starts when the engine fired and can hold the end of the wake word.
    - ``assistant``: ``assist`` (a Home Assistant Assist pipeline) or ``realtime``.
    - ``language``: the kiosk's language, for how the wake word is said there.
    - ``interaction``: how the turn started. ``wake_word``, ``question`` (a
      reply to something Home Assistant asked), ``follow_up`` (the
      conversation goes on) or ``manual`` (started by hand, with no wake word).
      ``ask_question`` and ``start_conversation`` reach a kiosk as the same
      announcement, so its start says ``question`` for both. It says which
      one in an ``interaction`` message right after, when Home Assistant's
      pipeline starts: one that will answer is a ``start_conversation``, one
      that ends at speech to text an ``ask_question``.
    """
    out: dict[str, Any] = {}
    for key in WAKE_TEXT:
        value = start.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = " ".join(value.split())[:80]
    if isinstance(start.get("seamless"), bool):
        out["seamless"] = start["seamless"]
    if start.get("assistant") in ("assist", "realtime"):
        out["assistant"] = start["assistant"]
    if start.get("interaction") in INTERACTIONS:
        out["interaction"] = start["interaction"]
    return out


async def session_ws(request: web.Request) -> web.WebSocketResponse:
    vm = request.app[VM]
    ws, start = await _kiosk_start(request)
    if start is None:
        return ws
    mode = start.get("mode", "live")
    policy = start.get("policy", "anyone")
    people = start.get("people") or []
    if (mode not in ("live", "gated", "follow") or policy not in ("anyone", "enrolled", "people")
            or not isinstance(people, list) or not all(isinstance(p, str) for p in people)):
        await ws.close(code=4400, message=b"bad mode or policy")
        return ws
    people = tuple(people) if policy == "people" else ()
    wake = wake_details(start)
    if mode == "follow":
        anchor = None
        if isinstance(start.get("anchor"), str):
            try:
                anchor = pcm16_to_float(base64.b64decode(start["anchor"]))[: 3 * 16000]
            except ValueError:
                anchor = None
        speaker = start.get("speaker") if isinstance(start.get("speaker"), str) else None
        isolation = start.get("isolation") is not False
        # A follow-up has no wake word of its own: nothing to leave out.
        first = speaker is None and not start.get("continues")
        # A short answer is only for a reply to something Home Assistant
        # asked. A kiosk that doesn't say how the turn started: None.
        asked = wake["interaction"] in ASKED if "interaction" in wake else None
        device = start.get("device_id")
        known = vm.hub.kiosks.get(device) if isinstance(device, str) else None
        feats = (known.features or {}) if known else {}
        face = (lambda: vm.at_kiosk.get(device)) if feats.get("face_confirms_voice") and feats.get("face_recognition") \
            else None
        # The kiosk's own Wake Word Aligner switch. A kiosk that doesn't say
        # (an older Kiosk Satellite) has it on.
        align = start.get("wake_word_aligner") is not False
        session = await vm.run(vm.new_follow, policy, anchor, people, speaker, isolation, wake if first else None,
                               asked, face, align)
    else:
        session = vm.new_session(mode=mode, policy=policy, people=people)
    device_id = start.get("device_id")
    # A follow-up in a conversation joins the session of its first turn.
    joins = start.get("continues") if mode == "follow" else None
    joins = joins if await vm.run(vm.sessions.continuing, joins, device_id) else None
    sid = joins or new_id()
    kiosk = vm.hub.seen(device_id, start.get("device_name"))
    if kiosk:
        kiosk.sessions += 1
        kiosk.in_session = True
    vm.status.sessions += 1
    vm.status.sessions_total += 1
    started = time.time()
    kept: list[dict[str, Any]] = []
    who = kiosk.name if kiosk else device_id or "an unknown kiosk"
    _LOG.info("session %s from %s: %s mode, answers %s%s%s%s", sid, who, mode, policy,
              f" ({len(people)} chosen)" if policy == "people" else "", ", a follow-up" if joins else "",
              _wake_text(wake))
    if policy == "people":
        _check_people(vm, device_id, people)
    vm.hub.publish({"type": "session_start", "session": sid, "device_id": device_id, "mode": mode})

    async def send(events: list[dict[str, Any]]) -> None:
        for e in events:
            if e["type"] == "audio":
                if not ws.closed:
                    await ws.send_bytes(float_to_pcm16(e["audio"]))
                continue
            if e["type"] == "keepalive":
                # For the kiosk only: not part of the session's record or the live view.
                if not ws.closed:
                    await ws.send_json(e)
                continue
            kept.append(e)
            if e["type"] != "stats":
                _LOG.debug("session %s: %s", sid, _event_text(e))
            vm.hub.publish({**e, "session": sid, "device_id": device_id})
            if not ws.closed:
                await ws.send_json(e)

    published = 0

    def live_scores() -> None:
        nonlocal published
        for item in session.timeline[published:]:
            vm.hub.publish({**item, "type": "score", "kind": item["kind"], "session": sid, "device_id": device_id})
        published = len(session.timeline)

    try:
        await ws.send_json({"type": "ready", "session": sid, "model": vm.model.key,
                            "threshold": vm.model.threshold})
        if mode == "follow":
            _LOG.debug("session %s: wake word anchor %s", sid,
                       f"{len(anchor) / 16000:.1f} s" if anchor is not None else "none")
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                await send(await vm.run(session.feed, pcm16_to_float(msg.data)))
                live_scores()
            elif msg.type == WSMsgType.TEXT:
                told = json.loads(msg.data)
                kind = told.get("type")
                if kind == "turn" and mode != "follow":
                    await send(session.new_turn())
                elif kind == "interaction" and told.get("interaction") in INTERACTIONS:
                    # Which kind of question it was, now that the kiosk knows.
                    wake["interaction"] = told["interaction"]
                elif kind == "end":
                    break
            else:
                break
    except Exception:
        _LOG.exception("session %s from %s failed", sid, who)
        raise
    finally:
        vm.status.sessions -= 1
        if kiosk:
            kiosk.in_session = False
        await send(await vm.run(session.finish))
        stats = next((e for e in kept if e["type"] == "stats"), {})
        _LOG.info("session %s from %s ended: %s", sid, who, _session_text(stats, kept))
        record = {"id": sid, "started": started, "device_id": device_id,
                  "device_name": kiosk.name if kiosk else None, "mode": mode, "policy": policy, "people": list(people),
                  "model": vm.model.key, "threshold": vm.model.threshold,
                  "other_threshold": vm.model.other_threshold,
                  "events": [e for e in kept if e["type"] != "stats"], "timeline": session.timeline,
                  "stats": stats}
        if wake:
            record["wake"] = wake
        if mode == "follow":
            # What went to the assistant, in seconds of what the kiosk sent.
            record["released"] = [[round(a / 16000, 3), round(b / 16000, 3)] for a, b in session.released]
            if session.lead_in_s:
                # Silence at the start of the sent audio that isn't in what the kiosk sent.
                record["lead_in_s"] = session.lead_in_s
        if joins:
            await vm.run(vm.sessions.append, sid, record, session.raw_audio(), session.cleaned_audio())
        else:
            await vm.run(vm.sessions.save, record, session.raw_audio(), session.cleaned_audio(),
                         getattr(session, "anchor_audio", None))
        vm.hub.publish({"type": "session_end", "session": sid, "device_id": device_id, "stats": stats})
        if not ws.closed:
            await ws.close()
    return ws


_LIST_FILTERS = ("kind", "device_id", "device", "person", "show", "q")


async def sessions_list(request: web.Request) -> web.Response:
    """A page of the Sessions list, newest first, with how many match in all.

    Filters: ``kind`` (voice or vision), ``device_id``, ``device`` (a kiosk by
    the name the list shows), ``person`` (a person's id or ``unknown``),
    ``show`` (``cut`` or ``rej``), ``q`` (text in the kiosk or speaker) and
    ``since`` (seconds since the epoch)."""
    vm = request.app[VM]
    q = request.query
    try:
        limit = max(1, min(int(q.get("limit", 100)), 500))
        offset = max(0, int(q.get("offset", 0)))
        since = float(q["since"]) if q.get("since") else None
    except ValueError as err:
        raise web.HTTPBadRequest(text="limit, offset and since must be numbers") from err
    filters = {k: q[k] for k in _LIST_FILTERS if q.get(k)}
    if filters.get("kind") not in (None, "voice", "vision"):
        raise web.HTTPBadRequest(text="kind must be voice or vision")

    def page() -> dict[str, Any]:
        return {"sessions": vm.sessions.list(limit, offset=offset, since=since, **filters),
                "total": vm.sessions.count(since=since, **filters), "offset": offset,
                "kiosks": vm.sessions.kiosks()}

    return web.json_response(await vm.run(page))


async def sessions_counts(request: web.Request) -> web.Response:
    """Sessions in the last ``hours`` (24 by default) by kind and outcome."""
    vm = request.app[VM]
    try:
        hours = max(1.0, min(float(request.query.get("hours", 24)), 24 * 31))
    except ValueError as err:
        raise web.HTTPBadRequest(text="hours must be a number") from err
    return web.json_response(await vm.run(vm.sessions.counts, time.time() - hours * 3600))


async def session_get(request: web.Request) -> web.Response:
    record = request.app[VM].sessions.get(request.match_info["sid"])
    if record is None:
        raise web.HTTPNotFound(text="no such session")
    return web.json_response(record)


async def session_audio(request: web.Request) -> web.StreamResponse:
    path = request.app[VM].sessions.audio_path(request.match_info["sid"], request.match_info["kind"])
    if path is None:
        raise web.HTTPNotFound(text="no audio for that session")
    return web.FileResponse(path, headers={"Content-Type": "audio/wav"})


async def session_frame(request: web.Request) -> web.StreamResponse:
    path = request.app[VM].sessions.frame_path(request.match_info["sid"], request.match_info["n"])
    if path is None:
        raise web.HTTPNotFound(text="no such frame")
    return web.FileResponse(path, headers={"Content-Type": "image/jpeg", "Cache-Control": "private, max-age=3600"})


async def frames_stats(request: web.Request) -> web.Response:
    """How many vision frames are saved and their size, for Settings."""
    vm = request.app[VM]
    return web.json_response(await vm.run(vm.sessions.frames_stats))


async def frames_delete(request: web.Request) -> web.Response:
    """Every saved vision frame. The sessions stay, without their frames."""
    vm = request.app[VM]
    gone = await vm.run(vm.sessions.delete_frames)
    _LOG.info("deleted %d vision frames (%.1f MB)", gone["frames"], gone["bytes"] / 1e6)
    return web.json_response(gone)


async def vision_looks(request: web.Request) -> web.Response:
    """Each kiosk's looks, empty looks and vision sessions by day."""
    vm = request.app[VM]
    try:
        days = max(1, min(int(request.query.get("days", 7)), 31))
    except ValueError as err:
        raise web.HTTPBadRequest(text="days must be a number") from err
    return web.json_response(await vm.run(vm.sessions.looks, days, request.query.get("device_id")))


SESSION_FILES = (("raw", "as-heard.wav"), ("clean", "sent-to-assistant.wav"), ("wake", "wake-word.wav"))


def _session_zip(vm: GroundControl, record: dict[str, Any]) -> bytes:
    """Everything about one session for an issue report: its audio or frames,
    its record and the service's details with the log lines from while it ran."""
    import io
    import zipfile

    sid = record["id"]
    start = float(record.get("started") or 0.0) - 2.0
    # The record is written when its last turn ends, follow-ups included.
    end = (vm.sessions.saved_at(sid) or record.get("saved_at") or start) + 5.0
    lines = [log_line(e) for e in LOGS.since(0, "DEBUG") if start <= e["t"] <= end]
    text = [*_report_header(vm), "", f"Session {sid}, log lines from while it ran:" if lines
            else f"Session {sid}: the log from while it ran is gone (the service restarted since).", *lines]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps(record, indent=1))
        z.writestr("ground-control.txt", "\n".join(text) + "\n")
        for kind, name in SESSION_FILES:
            path = vm.sessions.audio_path(sid, kind)
            if path is not None:
                z.write(path, name)
        for frame in record.get("frames") or []:
            path = vm.sessions.frame_path(sid, frame.get("n"))
            if path is not None:
                z.write(path, f"frames/look-{int(frame['n']):04d}-{frame.get('reason', 'look')}.jpg")
    return buf.getvalue()


async def session_download(request: web.Request) -> web.Response:
    """One zip with the session's audio or frames, its record and the log from while it ran."""
    vm = request.app[VM]
    record = vm.sessions.get(request.match_info["sid"])
    if record is None:
        raise web.HTTPNotFound(text="no such session")
    body = await vm.run(_session_zip, vm, record)
    return web.Response(body=body, content_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="ground-control-session-{record["id"]}.zip"'})


async def session_delete(request: web.Request) -> web.Response:
    if not request.app[VM].sessions.delete(request.match_info["sid"]):
        raise web.HTTPNotFound(text="no such session")
    return web.Response(status=204)


async def session_enroll(request: web.Request) -> web.Response:
    """Add a saved session's voice to a person: the cleaned audio, speech only."""
    vm = request.app[VM]
    sid = request.match_info["sid"]
    body = await _json(request)
    pid = body.get("person_id")
    if not pid or vm.people.get(pid) is None:
        raise web.HTTPBadRequest(text="person_id must name a person")
    path = vm.sessions.audio_path(sid, "clean") or vm.sessions.audio_path(sid, "raw")
    if path is None:
        raise web.HTTPNotFound(text="no audio for that session")
    record = vm.sessions.get(sid) or {}
    source = _source(vm, record.get("device_id"), record.get("device_name"))
    try:
        seconds = await vm.run(vm.people.enroll, pid, read_wav(path), vm.enroll_models(), source)
    except ValueError as err:
        raise web.HTTPBadRequest(text=str(err)) from err
    _LOG.info("added session %s to %s: %.1f s of speech", sid, vm.people.get(pid).name, seconds)
    return web.json_response({"added_s": round(seconds, 1), **_person(vm, vm.people.get(pid))})


# ── enrollment ────────────────────────────────────────────────────────────

def _passage(value: Any) -> list[str] | None:
    """The sentences the kiosk shows while the person reads, so the app shows
    the same text. Anything that isn't a short list of sentences is ignored."""
    if not isinstance(value, list) or not 0 < len(value) <= 40:
        return None
    if not all(isinstance(v, str) and 0 < len(v.strip()) <= 400 for v in value):
        return None
    return [v.strip() for v in value]


# ── vision ────────────────────────────────────────────────────────────────

VISION_WANTS = ("face", "hands")


def _vision_want(value: Any, default: tuple[str, ...]) -> tuple[str, ...] | None:
    """The looks a message asks for, or None when it asks for something unknown."""
    if value is None:
        return default
    if not isinstance(value, list) or not value or not all(v in VISION_WANTS for v in value):
        return None
    return tuple(dict.fromkeys(value))


async def vision_ws(request: web.Request) -> web.WebSocketResponse:
    """Vision Match: a kiosk's camera frames in, the faces and hands in them out.

    Each frame is a ``frame`` message and then its JPEG as one binary message.
    The kiosk waits for a frame's result before it sends the next one. A frame
    that arrives while another waits takes its place: the one it replaces is
    answered ``dropped``. ``outcome`` messages say what the kiosk did with the
    results, for the vision session they belong to (see
    ground_control.vision_sessions)."""
    vm = request.app[VM]
    ws, start = await _kiosk_start(request)
    if start is None:
        return ws
    want = _vision_want(start.get("want"), VISION_WANTS)
    if want is None:
        await ws.close(code=4400, message=b"bad want")
        return ws
    vision = await vm.run(vm.vision)
    if vision is None:
        await _refuse(request, ws, "vision_unavailable", vm.settings().get("vision_error") or "no vision models",
                      close_code=4503)
        return ws
    device_id = start.get("device_id")
    kiosk = vm.hub.seen(device_id, start.get("device_name"))
    who = kiosk.name if kiosk else device_id or "an unknown kiosk"
    recognize = start.get("recognize") is True and "face" in want
    _LOG.info("vision from %s: %s%s", who, " and ".join(want), " with Face Recognition" if recognize else "")
    await ws.send_json({"type": "ready", "model": vision.label, "models": vision.keys, "want": list(want),
                        "max_looks": vision.max_looks, "recognize": recognize})
    waiting: dict[str, Any] | None = None
    wake = asyncio.Event()
    looks = 0
    loop = asyncio.get_running_loop()
    saving: set[asyncio.Future] = set()

    def save(record: dict[str, Any], frames: list[tuple[int, bytes]]) -> None:
        job = loop.run_in_executor(vm.pool, lambda: vm.sessions.save(record, frames=frames))
        saving.add(job)
        job.add_done_callback(saved)

    def saved(job: asyncio.Future) -> None:
        saving.discard(job)
        if not job.cancelled() and job.exception() is not None:
            _LOG.error("could not save a vision session from %s: %s", who, job.exception())

    tracker = VisionTracker(device_id, kiosk.name if kiosk else start.get("device_name"), vision.label,
                            vision.on_gpu, keep_frames=lambda: vm.sessions.keep_frames, save=save,
                            publish=vm.hub.publish, models=vision.keys if vision.label == "mixed" else None)

    # Face Recognition: who is in front of this kiosk, over its face looks.
    rec = Recognizer(vm.face_threshold)
    prints: dict[str, Any] = {"ids": None}
    shown: dict[str, Any] = {"key": (None, True)}

    def faceprints() -> tuple[list[str], Any]:
        if prints["ids"] is None:
            prints["ids"], prints["m"] = vm.faces.faceprints(vision.rec_key)
        return prints["ids"], prints["m"]

    def forget_prints() -> None:
        prints["ids"] = None

    vm.faces.on_change.append(forget_prints)

    def person_of(match) -> dict[str, Any] | None:
        if match is None:
            return None
        p = vm.people.get(match.person_id) if match.person_id else None
        return {"id": p.id, "name": p.name, "color": p.color, "score": match.score} if p \
            else {"id": None, "name": None, "score": match.score}

    async def tell_person() -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """The person at the kiosk and everyone named, said once when either changes."""
        person = person_of(rec.current)
        people = [p for p in (person_of(m) for m in rec.people) if p and p["id"]]
        key = (person or {}).get("id"), person is None, tuple(p["id"] for p in people)
        if key != shown.get("key"):
            shown["key"] = key
            vm.at_kiosk[device_id] = (person or {}).get("id")
            tracker.person(person, people)
            vm.hub.publish({"type": "vision_person", "device_id": device_id, "person": person, "people": people})
            if person and person["id"]:
                others = [p["name"] for p in people if p["id"] != person["id"]]
                _LOG.info("%s is at %s (%.2f)%s", person["name"], who, person["score"],
                          f" with {', '.join(others)} also in view" if others else "")
        return person, people

    def flush() -> None:
        counts = tracker.take_counts()
        if any(counts.values()):
            loop.run_in_executor(vm.pool, vm.sessions.add_looks, device_id, counts)

    async def ticker() -> None:
        while True:
            await asyncio.sleep(1.0)
            if rec.tick(time.time()):
                person, people = await tell_person()
                if not ws.closed:
                    await ws.send_json({"type": "person", "person": person, "people": people})
            tracker.tick()
            flush()

    async def worker() -> None:
        nonlocal waiting, looks
        while True:
            await wake.wait()
            wake.clear()
            job, waiting = waiting, None
            if job is None:
                continue
            now = time.time()
            ids, _ = faceprints() if recognize else ([], None)
            embed = recognize and bool(ids) and "face" in job["want"] and rec.wants(now)
            try:
                look = await vm.run(vision.look, job["jpeg"], job["rotation"], job["want"], embed, vm.hand_min_size)
            except ValueError as err:
                tracker.lost("failed")
                if not ws.closed:
                    await ws.send_json({"type": "error", "id": job["id"], "code": "bad_frame", "message": str(err)})
                continue
            except Exception:  # noqa: BLE001 - the kiosk hears it and the next frame tries again
                _LOG.exception("vision look for %s failed", who)
                tracker.lost("failed")
                if not ws.closed:
                    await ws.send_json({"type": "error", "id": job["id"], "code": "look_failed"})
                continue
            looks += 1
            result: dict[str, Any] = {"type": "result", "id": job["id"], "ms": look.ms}
            if look.faces is not None:
                result["faces"] = look.faces
            if "hands" in job["want"]:
                result["hand"] = look.hand
            if recognize and look.faces is not None:
                facing = sorted((f["box"] for f in look.faces if f["frontal"]), key=lambda b: -b[2])
                if facing:
                    rec.seen(facing, now)
                if look.embeddings:
                    ids, m = faceprints()
                    # The Threshold as it is set now, not as it was when this kiosk connected.
                    rec.threshold = vm.face_threshold
                    rec.add(look.embeddings, ids, m, now)
                result["person"], result["people"] = await tell_person()
            _LOG.debug("vision %s frame %s: %s", who, job["id"], result)
            if not ws.closed:
                await ws.send_json(result)
            tracker.look(job["id"], job["want"], look.faces, look.hand if "hands" in job["want"] else None,
                         look.ms, job["jpeg"], rotation=job["rotation"], encode_ms=job.get("encode_ms"),
                         rtt_ms=job.get("rtt_ms"))

    task = asyncio.create_task(worker())
    ticking = asyncio.create_task(ticker())
    header: dict[str, Any] | None = None
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("type") == "outcome":
                    tracker.outcome(data)
                    continue
                if data.get("type") != "frame":
                    continue
                frame_want = _vision_want(data.get("want"), want)
                if frame_want is None or not set(frame_want) <= set(want):
                    await ws.send_json({"type": "error", "id": data.get("id"), "code": "bad_want"})
                    continue
                rotation = data.get("rotation", 0)
                header = {"id": data.get("id"), "want": frame_want,
                          "rotation": rotation if rotation in (0, 90, 180, 270) else 0,
                          "encode_ms": _ms(data.get("encode_ms")), "rtt_ms": _ms(data.get("rtt_ms"))}
            elif msg.type == WSMsgType.BINARY:
                if header is None:
                    continue  # a JPEG with no frame message before it
                header["jpeg"] = msg.data
                if waiting is not None and not ws.closed:
                    tracker.lost("dropped")
                    await ws.send_json({"type": "dropped", "id": waiting["id"]})
                waiting, header = header, None
                wake.set()
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        task.cancel()
        ticking.cancel()
        if forget_prints in vm.faces.on_change:
            vm.faces.on_change.remove(forget_prints)
        vm.at_kiosk.pop(device_id, None)
        tracker.tick()
        tracker.close("closed")
        flush()
        _LOG.info("vision from %s ended after %d looks", who, looks)
    return ws


def _ms(value: Any) -> float | None:
    """A time in milliseconds from a kiosk, or None when it isn't one."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value < 60000:
        return float(value)
    return None


async def face_enroll_ws(request: web.Request) -> web.WebSocketResponse:
    """Face Recognition's guided capture, through a kiosk's camera.

    Starts like ``/api/vision`` with ``person_id`` or a new person's ``name``.
    Each frame is a ``frame`` message and its JPEG, and gets a ``progress``
    answer with the step, the crops it has and needs and a hint for the
    person. When every step is done the crops are saved: ``saved``."""
    vm = request.app[VM]
    ws, start = await _kiosk_start(request)
    if start is None:
        return ws
    vision = await vm.run(vm.vision)
    if vision is None:
        await _refuse(request, ws, "vision_unavailable", vm.settings().get("vision_error") or "no vision models",
                      close_code=4503)
        return ws
    pid = start.get("person_id")
    try:
        if not pid:
            pid = vm.people.add(str(start.get("name", ""))).id
        elif vm.people.get(pid) is None:
            raise ValueError("no such person")
    except (ValueError, TypeError) as err:
        await ws.send_json({"type": "error", "code": "bad_request", "message": str(err)})
        await ws.close(code=4400)
        return ws
    from .faces import STEPS, FaceEnrollment
    from .vision import decode

    person = vm.people.get(pid)
    device_id = start.get("device_id")
    guide = FaceEnrollment(vision)
    kiosk = vm.hub.kiosks.get(device_id) if device_id else None
    where = kiosk.name if kiosk else start.get("device_name") or device_id or "a kiosk"

    def live(event: dict[str, Any]) -> None:
        vm.hub.publish({**event, "type": f"face_enroll_{event['type']}", "person_id": pid, "name": person.name,
                        "device_id": device_id})

    await ws.send_json({"type": "ready", "person_id": pid, "name": person.name,
                        "steps": [{"step": name, "need": need} for name, need in STEPS]})
    live({"type": "start", "steps": [{"step": name, "need": need} for name, need in STEPS]})
    _LOG.info("enrolling the face of %s on %s", person.name, where)

    def look(data: bytes, rotation: int) -> dict[str, Any]:
        rgb = decode(data, rotation)
        return guide.feed(rgb, vision.face(rgb), time.time())

    header: dict[str, Any] | None = None
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("type") == "end":
                break
            if data.get("type") == "frame":
                rotation = data.get("rotation", 0)
                header = {"id": data.get("id"), "rotation": rotation if rotation in (0, 90, 180, 270) else 0}
        elif msg.type == WSMsgType.BINARY:
            if header is None:
                continue
            frame, header = header, None
            try:
                event = await vm.run(look, msg.data, frame["rotation"])
            except ValueError as err:
                await ws.send_json({"type": "error", "id": frame["id"], "code": "bad_frame", "message": str(err)})
                continue
            event["id"] = frame["id"]
            await ws.send_json(event)
            live(event)
            if guide.done:
                break
        else:
            break
    if guide.done or len(guide.crops[STEPS[0][0]]) >= STEPS[0][1]:
        crops = guide.all_crops()
        try:
            source = _source(vm, device_id, start.get("device_name"), start.get("model"))
            record = await vm.run(vm.faces.enroll, pid, crops, vm.face_embedders(), source)
            saved: dict[str, Any] = {"type": "saved", "crops": len(crops), "enrollment": record["id"],
                                     "complete": guide.done}
        except ValueError as err:
            saved = {"type": "error", "code": "not_saved", "message": str(err)}
    else:
        saved = {"type": "error", "code": "too_short", "message": "the face wasn't seen long enough, nothing was saved"}
    live(saved)
    if saved["type"] == "saved":
        _LOG.info("face enrollment for %s saved: %d crops%s", person.name, saved["crops"],
                  "" if saved["complete"] else ", before every step was done")
    else:
        _LOG.warning("face enrollment for %s not saved: %s", person.name, saved["message"])
    if not ws.closed:
        await ws.send_json(saved)
        await ws.close()
    return ws


async def enroll_ws(request: web.Request) -> web.WebSocketResponse:
    vm = request.app[VM]
    ws, start = await _kiosk_start(request)
    if start is None:
        return ws
    pid = start.get("person_id")
    try:
        target = float(start.get("target_s", 45.0))
        if not 10.0 <= target <= 300.0:
            raise ValueError("target_s must be 10 to 300 seconds")
        if not pid:
            pid = vm.people.add(str(start.get("name", ""))).id
        elif vm.people.get(pid) is None:
            raise ValueError("no such person")
    except (ValueError, TypeError) as err:
        await ws.send_json({"type": "error", "code": "bad_request", "message": str(err)})
        await ws.close(code=4400)
        return ws
    person = vm.people.get(pid)
    device_id = start.get("device_id")
    passage = _passage(start.get("passage"))
    guide = GuidedEnrollment(vm.active, vm.vad, vm.model, target_s=target, passage=passage)
    await ws.send_json({"type": "ready", "person_id": pid, "name": person.name, "target_s": target})

    def live(event: dict[str, Any]) -> None:
        vm.hub.publish({**event, "type": f"enroll_{event['type']}", "person_id": pid, "name": person.name,
                        "device_id": device_id})

    live({"type": "start", "target_s": target, "passage": guide.passage, "sentence": 0})
    kiosk = vm.hub.kiosks.get(device_id) if device_id else None
    where = kiosk.name if kiosk else start.get("device_name") or device_id or "the browser"
    _LOG.info("enrolling %s on %s, %.0f s of speech to go", person.name, where, target)
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            for e in await vm.run(guide.feed, pcm16_to_float(msg.data)):
                live(e)
                await ws.send_json(e)
            if guide.done:
                break
        elif msg.type == WSMsgType.TEXT and json.loads(msg.data).get("type") == "end":
            break
        elif msg.type != WSMsgType.TEXT:
            break
    if guide.done or guide.speech_s >= MIN_PARTIAL_ENROLL_S:
        try:
            source = _source(vm, device_id, start.get("device_name"), start.get("model"))
            added = await vm.run(vm.people.enroll, pid, guide.speech(), vm.enroll_models(), source)
            saved: dict[str, Any] = {"type": "saved", "added_s": round(added, 1),
                                     "speech_s": round(vm.people.speech_seconds(pid), 1),
                                     "dropped_s": round(guide.dropped_s, 1)}
        except ValueError as err:
            saved = {"type": "error", "code": "not_saved", "message": str(err)}
    else:
        saved = {"type": "error", "code": "too_short",
                 "message": f"only {guide.speech_s:.0f} s of speech, nothing was saved"}
    live(saved)
    if saved["type"] == "saved":
        _LOG.info("enrollment for %s saved: %.1f s added, %.1f s left out as another voice", person.name,
                  saved["added_s"], saved["dropped_s"])
    else:
        _LOG.warning("enrollment for %s not saved: %s", person.name, saved["message"])
    if not ws.closed:
        await ws.send_json(saved)
        await ws.close()
    return ws


# ── kiosks and live events ────────────────────────────────────────────────

async def kiosk_ws(request: web.Request) -> web.WebSocketResponse:
    vm = request.app[VM]
    nonce = challenge()
    ws, hello = await _open_socket(request, "hello", nonce)
    if hello is None:
        return ws
    device_id = str(hello.get("device_id") or "")
    if not device_id:
        await _refuse(request, ws, "no_device_id", "the kiosk sent no name to go by. Give it a Device name "
                      "or an mDNS name in Kiosk Satellite, under Device", close_code=4400)
        return ws
    if not request.app[CLIENTS].verify(nonce, device_id, hello.get("proof")):
        await _refuse(request, ws, "not_kiosk_satellite", f"{device_id} did not prove it is Kiosk Satellite")
        return ws
    k = vm.hub.register(device_id, str(hello.get("name") or device_id), str(hello.get("version") or ""), ws,
                        model=str(hello.get("model") or ""), enroll=hello.get("enroll") is True,
                        features=_features(hello.get("features")))
    _LOG.info("kiosk %s connected from %s: %s, Kiosk Satellite %s, enrollment %s%s", k.name, request.remote,
              k.model or "unknown model", k.version or "unknown version", "allowed" if k.enroll else "off",
              "" if k.access == "approved" else f", {'waiting for approval' if k.access == 'pending' else 'declined'}")
    _check_features(vm, device_id, k.features)
    try:
        await _send_access(request, k.device_id)
        async for msg in ws:
            if msg.type not in (WSMsgType.TEXT, WSMsgType.BINARY):
                break
            vm.hub.seen(device_id)
            if msg.type == WSMsgType.TEXT:
                _from_kiosk(vm, device_id, msg.data)
    finally:
        vm.hub.unregister(device_id, ws)
        _LOG.info("kiosk %s disconnected%s", k.name, f" ({ws.close_code})" if ws.close_code else "")
    return ws


async def _send_access(request: web.Request, device_id: str) -> None:
    """Tell a connected kiosk where it stands: a ticket and everyone, or wait."""
    vm = request.app[VM]
    k = vm.hub.kiosks.get(device_id)
    if k is None:
        return
    if k.access == "approved":
        ticket = vm.hub.issue_ticket(device_id)
        await vm.hub.send(device_id, {"type": "ticket", "ticket": ticket})
        await vm.hub.send(device_id, await _people_message(request))
    else:
        await vm.hub.send(device_id, {"type": k.access})


async def _people_message(request: web.Request) -> dict[str, Any]:
    return await _people(request.app[VM])


async def _people(vm: GroundControl) -> dict[str, Any]:
    """Who the kiosks can answer and what Ground Control runs, for their analytics."""
    people = await vm.run(vm.people.summary)
    for p in people:
        p["has_face"] = vm.faces.has_face(str(p["id"]))
    return {"type": "people", "people": people, "model": vm.model.key,
            "version": __version__, "app_path": vm.app_path,
            **describe_install(bool(vm.cfg.supervisor_token), vm.active.on_gpu, vm.gpu)}


def _from_kiosk(vm: GroundControl, device_id: str, data: str) -> None:
    try:
        msg = json.loads(data)
    except ValueError:
        return
    if not isinstance(msg, dict):
        return
    if msg.get("type") == "settings":
        before = vm.hub.kiosks.get(device_id)
        was = before.enroll if before else None
        features = _features(msg.get("features"))
        vm.hub.update(device_id, msg.get("enroll") is True, features)
        _check_features(vm, device_id, features)
        if was != (msg.get("enroll") is True):
            _LOG.info("kiosk %s: enrollment %s", _kiosk_name(vm, device_id), "allowed" if msg.get("enroll") is True else "off")
    elif msg.get("type") == "enroll_error":
        _LOG.warning("kiosk %s could not start an enrollment: %s %s", _kiosk_name(vm, device_id),
                     msg.get("code") or "refused", msg.get("message") or "")
        # The kiosk couldn't start: the web UI waiting on it hears why.
        vm.hub.publish({"type": "enroll_error", "person_id": msg.get("person_id"), "device_id": device_id,
                        "code": str(msg.get("code") or "refused"), "message": str(msg.get("message") or "")})


async def events_ws(request: web.Request) -> web.WebSocketResponse:
    vm = request.app[VM]
    ws, start = await _open_socket(request, "listen", web_ui=True)
    if start is None:
        return ws
    vm.hub.listen(ws)
    try:
        await ws.send_json({"type": "kiosks", "kiosks": [k.public() for k in vm.hub.kiosks.values()]})
        async for msg in ws:
            if msg.type not in (WSMsgType.TEXT, WSMsgType.BINARY):
                break
    finally:
        vm.hub.unlisten(ws)
    return ws


async def kiosks_list(request: web.Request) -> web.Response:
    return web.json_response({"kiosks": [k.public() for k in request.app[VM].hub.kiosks.values()]})


async def kiosk_access(request: web.Request) -> web.Response:
    """Approve or decline a kiosk, from the Kiosks page."""
    vm = request.app[VM]
    access = (await _json(request)).get("access")
    device_id = request.match_info["device_id"]
    k = vm.hub.set_access(device_id, access) if access in ("approved", "declined", "pending") else None
    if k is None:
        raise web.HTTPBadRequest(text="access must be approved, declined or pending and the kiosk must be known")
    _LOG.info("kiosk %s %s", k.name, {"approved": "approved", "declined": "declined", "pending": "waiting for approval"}[access])
    if k.connected:
        await _send_access(request, device_id)
    return web.json_response(k.public())


async def kiosk_enroll(request: web.Request) -> web.Response:
    vm = request.app[VM]
    body = await _json(request)
    person = vm.people.get(str(body.get("person_id", "")))
    if person is None:
        raise web.HTTPBadRequest(text="person_id must name a person")
    kiosk = vm.hub.kiosks.get(request.match_info["device_id"])
    if kiosk is None or kiosk.access != "approved":
        raise web.HTTPConflict(text="that kiosk isn't approved")
    face = body.get("kind") == "face"
    features = kiosk.features or {}
    if face and not (features.get("vision_match") and features.get("face_recognition")):
        raise web.HTTPConflict(text="Face Recognition is off on that kiosk")
    if not face and kiosk.connected and not kiosk.enroll:
        raise web.HTTPConflict(text="Allow enrollment here is off on that kiosk")
    message = {"type": "enroll_face", "person_id": person.id, "name": person.name, "color": person.color} if face \
        else {"type": "enroll", "person_id": person.id, "name": person.name, "color": person.color,
              "target_s": float(body.get("target_s", 45.0))}
    sent = await vm.hub.send(request.match_info["device_id"], message)
    if not sent:
        raise web.HTTPConflict(text="that kiosk isn't connected")
    _LOG.info("asked %s to enroll %s%s", _kiosk_name(vm, request.match_info["device_id"]), person.name,
              "'s face" if face else "")
    return web.json_response({"sent": True})


async def kiosk_enroll_stop(request: web.Request) -> web.Response:
    """Stop a face capture running on a kiosk. The kiosk keeps what it has
    when the first step was done, as its own Stop does."""
    vm = request.app[VM]
    device_id = request.match_info["device_id"]
    kiosk = vm.hub.kiosks.get(device_id)
    if kiosk is None or kiosk.access != "approved":
        raise web.HTTPConflict(text="that kiosk isn't approved")
    if not await vm.hub.send(device_id, {"type": "enroll_face_stop"}):
        raise web.HTTPConflict(text="that kiosk isn't connected")
    _LOG.info("asked %s to stop its face capture", _kiosk_name(vm, device_id))
    return web.json_response({"sent": True})


# ── people ────────────────────────────────────────────────────────────────

async def people_list(request: web.Request) -> web.Response:
    vm = request.app[VM]
    return web.json_response({"people": [_person(vm, p) for p in vm.people.all()]})


async def people_add(request: web.Request) -> web.Response:
    vm = request.app[VM]
    body = await _json(request)
    try:
        p = vm.people.add(str(body.get("name", "")), body.get("color"))
    except ValueError as err:
        raise web.HTTPBadRequest(text=str(err)) from err
    _LOG.info("added %s", p.name)
    return web.json_response(_person(vm, p), status=201)


async def people_update(request: web.Request) -> web.Response:
    vm = request.app[VM]
    pid = _known(vm, request)
    body = await _json(request)
    try:
        before = vm.people.get(pid).name
        p = vm.people.update(pid, name=body.get("name"), color=body.get("color"), things=body.get("things"))
    except ValueError as err:
        raise web.HTTPBadRequest(text=str(err)) from err
    changed = [k for k in ("name", "color", "things") if body.get(k) is not None]
    _LOG.info("changed %s%s: %s", before, f" (now {p.name})" if p.name != before else "", ", ".join(changed) or "nothing")
    return web.json_response(_person(vm, p))


async def people_forget(request: web.Request) -> web.Response:
    vm = request.app[VM]
    pid = _known(vm, request)
    name = vm.people.get(pid).name
    vm.people.forget(pid)
    vm.faces.forget(pid)
    _LOG.info("forgot %s with their voice and face", name)
    return web.Response(status=204)


async def people_enroll(request: web.Request) -> web.Response:
    vm = request.app[VM]
    pid = _known(vm, request)
    try:
        audio = read_wav(await request.read(), any_rate=True)
        source = {"device_id": None, "device_name": request.query.get("device_name") or "Uploaded file",
                  "model": request.query.get("model")}
        seconds = await vm.run(vm.people.enroll, pid, audio, vm.enroll_models(), source)
    except (ValueError, EOFError) as err:
        _LOG.warning("a file for %s was not enrolled: %s", vm.people.get(pid).name, err)
        raise web.HTTPBadRequest(text=str(err)) from err
    _LOG.info("enrolled %s from a file: %.1f s of speech", vm.people.get(pid).name, seconds)
    return web.json_response({"added_s": round(seconds, 1), **_person(vm, vm.people.get(pid))})


async def enrollment_delete(request: web.Request) -> web.Response:
    vm = request.app[VM]
    pid = _known(vm, request)
    try:
        await vm.run(vm.people.delete_enrollment, pid, request.match_info["eid"], vm.enroll_models())
    except KeyError as err:
        raise web.HTTPNotFound(text="no such enrollment") from err
    _LOG.info("deleted an enrollment of %s and rebuilt their voice", vm.people.get(pid).name)
    return web.json_response(_person(vm, vm.people.get(pid)))


async def face_enrollment_delete(request: web.Request) -> web.Response:
    vm = request.app[VM]
    pid = _known(vm, request)
    try:
        await vm.run(vm.faces.delete_enrollment, pid, request.match_info["eid"], vm.face_embedders())
    except KeyError as err:
        raise web.HTTPNotFound(text="no such face enrollment") from err
    _LOG.info("deleted a face enrollment of %s and rebuilt their face", vm.people.get(pid).name)
    return web.json_response(_person(vm, vm.people.get(pid)))


async def face_crop(request: web.Request) -> web.StreamResponse:
    """One enrolled face crop, for the person's page."""
    vm = request.app[VM]
    pid = _known(vm, request)
    path = vm.faces.crop_path(pid, request.match_info["eid"], request.match_info["name"])
    if path is None:
        raise web.HTTPNotFound(text="no such face crop")
    return web.FileResponse(path, headers={"Content-Type": "image/jpeg", "Cache-Control": "private, max-age=3600"})


# ── settings ──────────────────────────────────────────────────────────────

async def settings_get(request: web.Request) -> web.Response:
    return web.json_response(request.app[VM].settings())


async def settings_update(request: web.Request) -> web.Response:
    vm = request.app[VM]
    body = await _json(request)
    rest = {k: v for k, v in body.items() if k != "model"}
    if rest:
        try:
            await vm.run(vm.update_settings, rest)
        except (ValueError, TypeError) as err:
            raise web.HTTPBadRequest(text=str(err)) from err
        _LOG.info("settings changed: %s", ", ".join(f"{k} {v}" for k, v in rest.items()))
    model = body.get("model")
    if model is not None:
        if model not in SPEAKER_MODELS:
            raise web.HTTPBadRequest(text="unknown model")
        _LOG.info("switching the speaker model to %s", model)
        asyncio.get_running_loop().create_task(vm.switch_model(model))
        await asyncio.sleep(0)  # let the switch mark itself as rebuilding
    return await settings_get(request)


# ── logs ──────────────────────────────────────────────────────────────────

def _kiosk_name(vm: GroundControl, device_id: str | None) -> str:
    k = vm.hub.kiosks.get(device_id or "")
    return k.name if k else device_id or "an unknown kiosk"


def _event_text(e: dict[str, Any]) -> str:
    """A session event in a few words, for detailed logs."""
    rest = {k: v for k, v in e.items() if k not in ("type", "t") and v is not None}
    at = f" at {e['t']:.2f} s" if isinstance(e.get("t"), (int, float)) else ""
    return f"{e['type']}{at}" + (f" {json.dumps(rest, separators=(',', ':'))}" if rest else "")


def _wake_text(wake: dict[str, Any]) -> str:
    """What the kiosk said about how a session started, for its line in the log."""
    parts = []
    if wake.get("wake_word"):
        engine = f" ({wake['wake_word_engine']})" if wake.get("wake_word_engine") else ""
        parts.append(f'wake word "{wake["wake_word"]}"{engine}')
    if wake.get("interaction") in ("question", "follow_up", "manual"):
        parts.append({"question": "a reply to a question from Home Assistant", "follow_up": "a follow-up",
                      "manual": "started by hand"}[wake["interaction"]])
    elif wake.get("interaction") in ("ask_question", "start_conversation"):
        parts.append(f"a reply to {wake['interaction']}")
    if wake.get("seamless"):
        parts.append("talk right after the wake word")
    if wake.get("assistant"):
        parts.append("a realtime assistant" if wake["assistant"] == "realtime" else "an Assist pipeline")
    return "".join(f", {p}" for p in parts)


def _session_text(stats: dict[str, Any], events: list[dict[str, Any]]) -> str:
    """How a session went, in one line: who, why it ended and how much audio."""
    speaker = stats.get("speaker") or {}
    if stats.get("rejected"):
        who = (f"rejected {speaker['name']}, not one of the people this kiosk answers" if speaker.get("name")
               else "rejected an unknown voice")
    elif speaker.get("name"):
        score = f" {speaker['score']}" if speaker.get("score") is not None else ""
        who = f"{speaker['name']} ({speaker.get('by') or 'voice'}{score})"
    elif speaker.get("by") == "answer":
        who = f"a short answer taken ({speaker.get('score')}, needed {speaker.get('line')})"
    elif any(e["type"] == "speaker" for e in events):
        who = "an unknown voice"
    else:
        who = "nobody talked"
    parts = [who, f"ended by {stats.get('end') or 'the kiosk'}", f"{stats.get('audio_s', 0)} s of audio"]
    if stats.get("turns"):
        parts.append(f"{stats['turns']} turns, {stats.get('released_s', 0)} s to the assistant")
    if stats.get("dropped_s"):
        parts.append(f"{stats['dropped_s']} s of other voices left out")
    if stats.get("left_out"):
        n = stats["left_out"]
        parts.append(f"{n} stretch{'es' if n != 1 else ''} of another voice held and left out")
    if stats.get("kept_out"):
        n = stats["kept_out"]
        parts.append(f"another voice kept from starting a turn {n} time{'s' if n != 1 else ''}")
    if stats.get("isolation_skipped"):
        n = stats["isolation_skipped"]
        parts.append(f"Isolation not used for {n} turn{'s' if n != 1 else ''}, which it returned as near silence")
    if stats.get("isolation_failed"):
        n = stats["isolation_failed"]
        parts.append(f"Isolation not used for {n} turn{'s' if n != 1 else ''}, since the GPU didn't answer")
    if stats.get("wake_word_s"):
        parts.append(f"{stats['wake_word_s']} s of the wake word left out")
    if stats.get("cpu_ms") is not None:
        parts.append(f"{stats['cpu_ms']} ms of CPU")
    return ", ".join(parts)


async def logs_get(request: web.Request) -> web.Response:
    try:
        after = int(request.query.get("after", 0))
    except ValueError as err:
        raise web.HTTPBadRequest(text="after must be a number") from err
    entries = LOGS.since(after, request.query.get("level", "DEBUG"))[-2000:]
    return web.json_response({"entries": entries, "last": LOGS.last, "detailed": LOGS.detailed,
                              "started": LOGS.started})


async def logs_update(request: web.Request) -> web.Response:
    body = await _json(request)
    if not isinstance(body.get("detailed"), bool):
        raise web.HTTPBadRequest(text="detailed must be true or false")
    LOGS.detailed = body["detailed"]
    return web.json_response({"detailed": LOGS.detailed})


def _report_header(vm: GroundControl) -> list[str]:
    import platform

    s = vm.settings()
    kiosks = sorted(vm.hub.kiosks.values(), key=lambda k: (not k.connected, k.name.lower()))
    up = time.time() - LOGS.started
    lines = [
        f"Kiosk Satellite Ground Control {__version__}",
        f"Report made {time.strftime('%Y-%m-%d %H:%M:%S %z')}, running for {int(up // 3600)} h {int(up % 3600 // 60)} min",
        f"Runs {'as a Home Assistant app' if vm.cfg.supervisor_token else 'standalone'}"
        f" on {platform.system()} {platform.machine()}, Python {platform.python_version()}",
        f"Speaker model {vm.model.label} on {'the GPU' if vm.active.on_gpu else 'the CPU'}"
        f" ({vm.cfg.threads} threads, CUDA {'on' if vm.cfg.cuda else 'off'}"
        f"{', OpenVINO on ' + s['openvino'] if s.get('openvino') else ''}"
        f"{', Vulkan on ' + s['vulkan'] if s.get('vulkan') else ''}),"
        f" thresholds {vm.model.threshold} and {vm.model.other_threshold}",
        f"Settings: {json.dumps({k: v for k, v in s.items() if k not in ('models', 'default_threshold', 'default_other_threshold')}, separators=(', ', ': '))}",
        f"People: {len(vm.people.all())}, {sum(vm.people.has_voice(p.id) for p in vm.people.all())} with a voice",
        f"Detailed logs {'on' if LOGS.detailed else 'off'}",
        "Kiosks:" if kiosks else "Kiosks: none",
    ]
    for k in kiosks:
        state = "connected" if k.connected else f"last seen {time.strftime('%Y-%m-%d %H:%M', time.localtime(k.last_seen))}"
        lines.append(f"  {k.name}: {k.model or 'unknown model'}, Kiosk Satellite {k.version or '?'}, {state},"
                     f" {k.sessions} session{'' if k.sessions == 1 else 's'}, enrollment {'allowed' if k.enroll else 'off'}")
    return lines


async def logs_report(request: web.Request) -> web.Response:
    vm = request.app[VM]
    text = "\n".join([*_report_header(vm), "", *map(log_line, LOGS.since(0, request.query.get("level", "DEBUG")))])
    name = f"ground-control-logs-{time.strftime('%Y%m%d-%H%M')}.txt"
    return web.Response(text=text + "\n", content_type="text/plain", charset="utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})


ADVERTISER = web.AppKey("advertiser", object)


def app_path(slug: str, ha_version: str | None) -> str:
    """The app's page in Home Assistant. 2026.2 moved it from ``/hassio/ingress`` to ``/app`` (Kiosk Satellite #1563)."""
    try:
        year, month = (int(p) for p in (ha_version or "").split(".")[:2])
    except ValueError:
        return f"/app/{slug}"
    return f"/hassio/ingress/{slug}" if (year, month) < (2026, 2) else f"/app/{slug}"


async def _find_app_path(vm: GroundControl, supervisor_token: str) -> None:
    """As a Home Assistant app: where its page is, for the kiosks' Open."""
    import aiohttp

    headers = {"Authorization": f"Bearer {supervisor_token}"}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as http:
            async with http.get("http://supervisor/addons/self/info", headers=headers) as r:
                slug = (await r.json())["data"]["slug"]
            version = None
            try:
                async with http.get("http://supervisor/core/info", headers=headers) as r:
                    version = (await r.json())["data"]["version"]
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError, TypeError):
                pass  # a current Home Assistant is the likely one
        vm.app_path = app_path(slug, version)
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as err:
        _LOG.warning("could not read this app's slug from the Supervisor: %s", err)


async def _check_web_token(app: web.Application) -> None:
    """Say at the start whether ``HA_TOKEN`` opens the web UI."""
    vm = app[VM]
    ok = await app[AUTH].check(vm.cfg.ha_token)
    if ok:
        _LOG.warning("HA_TOKEN signs in the web UI: anyone who can reach port %s opens Ground Control "
                     "without logging in", vm.cfg.port)
    elif ok is False:
        _LOG.warning("Home Assistant refused HA_TOKEN: the web UI asks to log in with Home Assistant")
    else:
        _LOG.warning("could not reach Home Assistant at %s to check HA_TOKEN: the web UI waits for it",
                     app[AUTH].ha_url)


def build_app(vm: GroundControl, token_check: TokenCheck | None = None, mdns: bool = False,
              clients: ClientCheck | None = None) -> web.Application:
    app = web.Application(middlewares=[require_token], client_max_size=512 << 20)
    app[VM] = vm
    app[AUTH] = token_check or TokenCheck(vm.cfg.ha_url, supervisor_token=vm.cfg.supervisor_token)
    app[OPEN] = vm.cfg.auth == "none"
    app[CLIENTS] = clients or ClientCheck()
    app[PICTURES] = Pictures(vm.cfg.ha_url, vm.cfg.supervisor_token)
    app[SUGGESTIONS] = Suggestions(vm.cfg.ha_url, vm.cfg.supervisor_token)
    app[LOGIN] = HomeAssistantLogin(vm.cfg.ha_url)
    if vm.cfg.update_check:
        app[UPDATES] = UpdateCheck(__version__)
    r = app.router

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    r.add_get("/", index)
    r.add_static("/static/", WEB, append_version=False)
    r.add_get("/health", health)
    r.add_get("/auth/info", auth_info)
    r.add_get("/auth/login", auth_login)
    r.add_get("/auth/callback", auth_callback)
    r.add_post("/auth/refresh", auth_refresh)
    r.add_post("/auth/logout", auth_logout)
    r.add_get("/api/status", status)
    r.add_get("/api/session", session_ws)
    r.add_get("/api/enroll", enroll_ws)
    r.add_get("/api/face_enroll", face_enroll_ws)
    r.add_get("/api/kiosk", kiosk_ws)
    r.add_get("/api/vision", vision_ws)
    r.add_get("/api/events", events_ws)
    r.add_get("/api/people", people_list)
    r.add_get("/api/people/{id}/picture", person_picture)
    r.add_get("/api/suggestions", suggestions)
    r.add_post("/api/people", people_add)
    r.add_patch("/api/people/{id}", people_update)
    r.add_delete("/api/people/{id}", people_forget)
    r.add_post("/api/people/{id}/enroll", people_enroll)
    r.add_delete("/api/people/{id}/enrollments/{eid}", enrollment_delete)
    r.add_delete("/api/people/{id}/faces/{eid}", face_enrollment_delete)
    r.add_get("/api/people/{id}/faces/{eid}/{name}", face_crop)
    r.add_get("/api/sessions", sessions_list)
    r.add_get("/api/sessions/counts", sessions_counts)
    r.add_get("/api/sessions/{sid}", session_get)
    r.add_get("/api/sessions/{sid}/audio/{kind}", session_audio)
    r.add_get("/api/sessions/{sid}/frames/{n}", session_frame)
    r.add_get("/api/frames", frames_stats)
    r.add_delete("/api/frames", frames_delete)
    r.add_get("/api/vision/looks", vision_looks)
    r.add_get("/api/sessions/{sid}/download", session_download)
    r.add_post("/api/sessions/{sid}/enroll", session_enroll)
    r.add_delete("/api/sessions/{sid}", session_delete)
    r.add_get("/api/kiosks", kiosks_list)
    r.add_post("/api/kiosks/{device_id}/enroll", kiosk_enroll)
    r.add_post("/api/kiosks/{device_id}/enroll_stop", kiosk_enroll_stop)
    r.add_post("/api/kiosks/{device_id}/access", kiosk_access)
    r.add_get("/api/settings", settings_get)
    r.add_patch("/api/settings", settings_update)
    r.add_get("/api/logs", logs_get)
    r.add_patch("/api/logs", logs_update)
    r.add_get("/api/logs/report", logs_report)

    async def on_startup(app: web.Application) -> None:
        loop = asyncio.get_running_loop()

        def people_changed() -> None:
            async def push() -> None:
                await vm.hub.broadcast(await _people(vm))
            loop.call_soon_threadsafe(lambda: loop.create_task(push()))

        vm.people.on_change.append(people_changed)
        vm.faces.on_change.append(people_changed)
        if UPDATES in app:
            app[UPDATES].start()
        if vm.cfg.supervisor_token:
            await _find_app_path(vm, vm.cfg.supervisor_token)
        if vm.cfg.ha_token and not app[OPEN]:
            loop.create_task(_check_web_token(app))
        if mdns:
            from .mdns import Advertiser

            advertiser = Advertiser(vm.cfg.port, vm.model.key, vm.cfg.mdns_address)
            try:
                await advertiser.start()
                app[ADVERTISER] = advertiser
            except Exception as err:  # noqa: BLE001
                # Kiosks can still be given the address: never stop the service over it.
                _LOG.warning("mDNS advertising failed, kiosks need the address entered: %s", err)

    async def on_cleanup(app: web.Application) -> None:
        if ADVERTISER in app:
            await app[ADVERTISER].stop()
        await app[AUTH].close()
        await app[PICTURES].close()
        await app[SUGGESTIONS].close()
        await app[LOGIN].close()
        if UPDATES in app:
            await app[UPDATES].close()
        vm.close()

    async def revalidate(request: web.Request, response: web.StreamResponse) -> None:
        # The UI's files change with every update: make browsers check, not guess.
        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"

    app.on_response_prepare.append(revalidate)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app
