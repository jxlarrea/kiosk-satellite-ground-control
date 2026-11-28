from __future__ import annotations

import asyncio

import numpy as np
import pytest
from aiohttp import WSMsgType

from ground_control.audio import float_to_pcm16, wav_bytes
from ground_control.auth import TokenCheck
from ground_control.server import VM, build_app
from ground_control.service import GroundControl

from .conftest import SR, chunks

from .kiosks import CLIENTS, approve

GOOD = {"Authorization": "Bearer good"}


@pytest.fixture
async def client(aiohttp_client, cfg):
    async def ask(token):
        return token == "good"

    vm = GroundControl(cfg)
    return await aiohttp_client(build_app(vm, TokenCheck(cfg.ha_url, ask=ask), clients=CLIENTS))


async def test_health_needs_no_token(client):
    r = await client.get("/health")
    assert r.status == 200


async def test_api_needs_a_token(client):
    assert (await client.get("/api/people")).status == 401
    assert (await client.get("/api/people", headers={"Authorization": "Bearer bad"})).status == 401
    assert (await client.get("/api/people", headers=GOOD)).status == 200


async def test_session_refuses_a_bad_token(client):
    ws = await client.ws_connect("/api/session")
    await ws.send_json({"type": "start", "token": "bad"})
    msg = await ws.receive()
    assert msg.type == WSMsgType.TEXT and msg.json()["code"] == "unauthorized"
    msg = await ws.receive()
    assert msg.type == WSMsgType.CLOSE and ws.close_code == 4401


async def test_enroll_then_a_session(client, speakers, voice):
    r = await client.post("/api/people", json={"name": "Ana"}, headers=GOOD)
    assert r.status == 201
    ana = await r.json()
    r = await client.post(f"/api/people/{ana['id']}/enroll", data=wav_bytes(voice[speakers[0]]["enroll"]),
                          headers=GOOD)
    assert r.status == 200 and (await r.json())["speech_s"] > 30

    ws = await client.ws_connect("/api/session")
    await ws.send_json({"type": "start", "token": "good", "ticket": approve(client, "test"), "device_id": "test", "mode": "gated"})
    ready = await ws.receive_json()
    assert ready["type"] == "ready" and ready["model"] == "accurate"
    audio = np.concatenate([voice[speakers[0]]["test"][:4 * SR], voice[speakers[2]]["test"][:4 * SR]])
    for c in chunks(audio):
        await ws.send_bytes(float_to_pcm16(c))
    await ws.send_json({"type": "end"})
    events, gated = [], 0
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            gated += len(msg.data) // 2
        elif msg.type == WSMsgType.TEXT:
            events.append(msg.json())
    kinds = [e["type"] for e in events]
    assert "speaker" in kinds and "other_voice" in kinds and kinds[-1] == "stats"
    speaker = next(e for e in events if e["type"] == "speaker")
    assert speaker["name"] == "Ana"
    cut = next(e for e in events if e["type"] == "other_voice")["cut"]
    assert gated <= cut * SR + 1

    vm = client.app[VM]
    assert (await vm.run(vm.identify, voice[speakers[0]]["test"]))["name"] == "Ana"


async def test_people_api(client):
    r = await client.post("/api/people", json={"name": "Leo"}, headers=GOOD)
    leo = await r.json()
    r = await client.patch(f"/api/people/{leo['id']}", json={"things": {"todo_list": "todo.leo"}}, headers=GOOD)
    assert (await r.json())["things"] == {"todo_list": "todo.leo"}
    assert (await client.patch(f"/api/people/{leo['id']}", json={"things": {"car": "x"}},
                               headers=GOOD)).status == 400
    assert (await client.delete(f"/api/people/{leo['id']}", headers=GOOD)).status == 204
    assert (await client.delete(f"/api/people/{leo['id']}", headers=GOOD)).status == 404


async def test_model_switch_rebuilds_and_persists(client, cfg, speakers, voice):
    r = await client.post("/api/people", json={"name": "Ana"}, headers=GOOD)
    ana = await r.json()
    await client.post(f"/api/people/{ana['id']}/enroll", data=wav_bytes(voice[speakers[0]]["enroll"]), headers=GOOD)
    r = await client.patch("/api/settings", json={"model": "light"}, headers=GOOD)
    assert (await r.json())["rebuilding"] == "light"
    for _ in range(100):
        s = await (await client.get("/api/settings", headers=GOOD)).json()
        if s["rebuilding"] is None:
            break
        await asyncio.sleep(0.1)
    assert s["model"] == "light"
    vm = client.app[VM]
    assert (await vm.run(vm.identify, voice[speakers[0]]["test"]))["name"] == "Ana"
    assert GroundControl(cfg).status.model == "light"


async def test_ui_loads_without_a_token(client):
    r = await client.get("/")
    assert r.status == 200 and "static/app.js" in await r.text()
    r = await client.get("/static/app.js")
    assert r.status == 200 and r.headers["Cache-Control"] == "no-cache"
    assert (await client.get("/static/pages/people.js")).status == 200


def test_ingress_is_trusted_only_from_the_supervisor():
    from aiohttp.test_utils import make_mocked_request
    from unittest.mock import MagicMock

    from ground_control.server import _from_ingress

    def req(ip, headers):
        transport = MagicMock()
        transport.get_extra_info.return_value = (ip, 1234)
        return make_mocked_request("GET", "/api/people", headers=headers, transport=transport)

    assert _from_ingress(req("172.30.32.2", {"X-Ingress-Path": "/api/hassio_ingress/abc"}))
    assert not _from_ingress(req("172.30.32.2", {}))
    assert not _from_ingress(req("192.168.1.5", {"X-Ingress-Path": "/api/hassio_ingress/abc"}))


async def test_settings_are_checked_and_kept(client, cfg):
    r = await client.patch("/api/settings", json={"threshold": 0.55, "other_threshold": 0.3,
                                                  "keep_days": 30, "keep_audio": False}, headers=GOOD)
    s = await r.json()
    assert (s["threshold"], s["other_threshold"], s["keep_days"], s["keep_audio"]) == (0.55, 0.3, 30, False)
    assert "learn" not in s
    assert (await client.patch("/api/settings", json={"other_threshold": 0.6}, headers=GOOD)).status == 400
    assert (await client.patch("/api/settings", json={"keep_days": 0}, headers=GOOD)).status == 400
    again = GroundControl(cfg)
    assert again.model.threshold == 0.55 and again.sessions.keep_days == 30


def test_the_mdns_service_type_is_valid():
    """mDNS caps a service name at 15 bytes. A longer one stopped the service at startup."""
    from zeroconf._utils.name import service_type_name

    from ground_control.mdns import SERVICE_TYPE

    assert service_type_name(SERVICE_TYPE) == SERVICE_TYPE


def test_the_default_speaker_model_fits_the_machine():
    from ground_control.models import default_model_key

    assert default_model_key("x86_64", 20) == "accurate"  # an i9
    assert default_model_key("x86_64", 4) == "light"  # an Intel N100
    assert default_model_key("aarch64", 4) == "light"  # a Raspberry Pi
    assert default_model_key("aarch64", 20, cuda=True) == "accurate"  # an NVIDIA GB10
    assert default_model_key("x86_64", 4, cuda=True) == "accurate"
    assert default_model_key("aarch64", 8, apple=True) == "accurate"  # a Mac, in Docker's VM


def test_apple_silicon_is_found_from_the_cpu_implementer(tmp_path):
    from ground_control.models import apple_silicon

    cpu = tmp_path / "cpuinfo"
    cpu.write_text("processor\t: 0\nBogoMIPS\t: 48.00\nCPU implementer\t: 0x61\nCPU architecture: 8\n")
    assert apple_silicon(cpu)
    cpu.write_text("processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd08\n")  # a Raspberry Pi 4
    assert not apple_silicon(cpu)
    cpu.write_text("processor\t: 0\nvendor_id\t: GenuineIntel\n")
    assert not apple_silicon(cpu)
    assert not apple_silicon(tmp_path / "missing")
