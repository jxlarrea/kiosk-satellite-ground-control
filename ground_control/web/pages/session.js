import { api, audioBytes, audioUrl } from "../api.js";
import { shell } from "../app.js";
import { loudness } from "../loudness.js";
import { analyze, draw } from "../spectrogram.js";
import { answersLabel, ask, chip, clock, h, icon, inform, modeLabel, outcome, outcomeChip, secs, section, toast } from "../ui.js";
import { renderVision } from "./vision_session.js";

const BARS = 104;

function pcm(buffer) {
  // Our WAVs are 16 kHz mono PCM16 with a 44-byte header.
  return new Int16Array(buffer.slice(44));
}

/** Follow mode: the stretches that went to the assistant in the speaker's
 * color, everything left out in the other voice's. */
function followWave(samples, total, released, color) {
  const per = Math.max(1, Math.floor(samples.length / BARS));
  const bars = [];
  for (let i = 0; i < BARS; i++) {
    let peak = 0;
    for (let j = i * per; j < Math.min(samples.length, (i + 1) * per); j++) peak = Math.max(peak, Math.abs(samples[j]));
    const t = ((i + 0.5) / BARS) * total;
    const sent = released.some(([a, b]) => t >= a && t < b);
    bars.push(h("span", { style: { height: `${Math.max(3, Math.round((peak / 32768) * 72))}px`,
      background: sent ? color : "var(--other)", opacity: sent ? 1 : 0.45 } }));
  }
  return bars;
}

function waveform(samples, total, events, color) {
  const start = events.find((e) => e.type === "speech_start")?.t ?? 0;
  const cut = events.find((e) => e.type === "other_voice")?.cut;
  const end = events.find((e) => e.type === "end_of_speech")?.t;
  const per = Math.max(1, Math.floor(samples.length / BARS));
  const bars = [];
  for (let i = 0; i < BARS; i++) {
    let peak = 0;
    for (let j = i * per; j < Math.min(samples.length, (i + 1) * per); j++) peak = Math.max(peak, Math.abs(samples[j]));
    const t = ((i + 0.5) / BARS) * total;
    let c = color, o = 1;
    if (t < start) o = 0.55;
    if (cut !== undefined && t >= cut) c = "var(--other)";
    if (end !== undefined && t > end) o = 0.3;
    bars.push(h("span", { style: { height: `${Math.max(3, Math.round((peak / 32768) * 72))}px`, background: c, opacity: o } }));
  }
  return bars;
}

// What each point on the chart is. Who is speaking is judged against the
// Recognized line, whether it is still the user's voice against Other voice.
const POINT = {
  anchor: "The wake word: who said it",
  identify: "Who is speaking",
  check: "Still the user's voice",
  word: "Still the user's voice",
};

function chart(record, total, color) {
  const pts = (record.timeline || []).filter((p) => p.t !== undefined && p.score !== undefined && p.kind !== "face"
    && (record.mode === "follow" || p.turn === 0 || p.turn === undefined)).map((p) => {
    const who = p.kind === "anchor" || p.kind === "identify";
    const label = p.kind === "identify" && p.isolated ? "Who is speaking, scored on the command after Isolation"
      : p.kind === "identify" && p.room ? "Who is speaking, scored on the turn and the room around it"
      : p.kind === "identify" && p.wake_word ? "Who is speaking, scored on the wake word and the command"
      : p.kind === "check" && p.low ? "Not the user's voice" : POINT[p.kind] || "Check";
    return h("span", {
      class: `pt ${who ? "who" : ""}`,
      title: `${label} at ${p.t} s: ${p.score}`,
      style: {
        left: `${(p.t / total) * 100}%`, bottom: `${Math.max(0, Math.min(1, p.score)) * 100}%`,
        background: p.kind === "check" && p.low ? "var(--error)" : who ? "var(--primary)" : color,
      },
    });
  });
  return h("div", { class: "chart", role: "img", "aria-label": "Voice score over time" },
    h("span", { class: "yl", style: { bottom: "100%" } }, "1.0"),
    h("span", { class: "yl", style: { bottom: "0%" } }, "0"),
    h("div", { class: "thr", style: { bottom: `${record.threshold * 100}%` } }, h("span", { class: "thrlbl" }, `Recognized ${record.threshold}`)),
    h("div", { class: "thr", style: { bottom: `${record.other_threshold * 100}%` } }, h("span", { class: "thrlbl" }, `Other voice ${record.other_threshold}`)),
    pts);
}

function chartKey(record, color) {
  const low = (record.timeline || []).some((p) => p.kind === "check" && p.low);
  return h("div", { class: "chartkey" },
    h("span", {}, h("span", { class: "pt who key", style: { background: "var(--primary)" } }), "Who is speaking, compared with the Recognized line"),
    h("span", {}, h("span", { class: "pt key", style: { background: color } }), "Still the user's voice, compared with the Other voice line"),
    low ? h("span", {}, h("span", { class: "pt key", style: { background: "var(--error)" } }), "Under the Other voice line: not the user and left out") : null);
}

function steps(record) {
  // The end of the wake word first: the user's own first word can start right at it.
  const out = (record.timeline || []).filter((p) => p.kind === "wake_word").map((p) => wakeWordStep(p, record.wake?.wake_word));
  out.push(...(record.followups || []).map((t) => ["Follow-up", "The assistant asked something and the user answered. This part is the answer.", t]));
  for (const e of record.events || []) {
    if (e.type === "speech_start") out.push(["Speech started", "The voice check starts here.", e.t]);
    if (e.type === "speaker" && e.name) {
      const both = (record.timeline || []).some((p) => p.kind === "identify" && p.wake_word && p.score === e.score);
      const isolated = (record.timeline || []).some((p) => p.kind === "identify" && p.isolated && p.score === e.score);
      out.push([`${e.name} recognized`, e.by === "only_person" ? "The only person enrolled. There was no voice to compare."
        : e.by === "conversation" ? "Recognized earlier in this conversation. Only their voice goes on."
        : e.by === "face" ? `Score ${e.score} is under the threshold. ${e.name}'s face was recognized at the kiosk, so the voice was taken for them.`
        : isolated ? `Score ${e.score} on the command after Isolation. With another voice over it, the command as the kiosk heard it was under the threshold.`
        : both ? `Score ${e.score} from the wake word and the command together. The command alone was too short.` : `Score ${e.score}.`, e.t]);
    }
    if (e.type === "speaker" && !e.name && e.by === "answer") {
      out.push(["Short answer taken", `The answer was scored together with the room around it: ${e.score} against the closest of the people this kiosk answers. With no wake word, an answer this short needs ${e.line}. It is too short to say who is speaking.`, e.t]);
    } else if (e.type === "speaker" && !e.name) out.push(["Unknown voice", `The best score of ${e.score} is under the threshold.`, e.t]);
    if (e.type === "other_voice" && record.mode === "follow") {
      out.push(["Another voice left out", "It never reached the assistant.", e.t]);
    } else if (e.type === "other_voice") {
      out.push(["Speaker's last word", "Everything after it was left out.", e.cut]);
      out.push(["Another voice", `Two checks in a row under ${record.other_threshold}.`, e.t]);
    }
    if (e.type === "turn_start") out.push([e.turn > 1 ? `Turn ${e.turn}` : "The user started talking", "What they say goes to the assistant.", e.t]);
    if (e.type === "turn_end") out.push(["The user stopped", `${e.speech_s ?? 0} s of speech. The command ended here.`, e.t]);
    if (e.type === "rejected") out.push(["Rejected", "This kiosk answers enrolled voices only.", e.t]);
    if (e.type === "end_of_speech" && e.reason === "silence") out.push(["Speaker stopped", "Silence after the speech.", e.t]);
  }
  for (const p of record.timeline || []) {
    if (p.kind === "voices" && p.kept_out) out.push(["Another voice kept out", `Speech here could have started a new turn by the voiceprint alone. Compared with the voices of this session, it was closer to the other voice heard in it (${p.other}) than to the user as heard in it (${p.user}), so no turn started.`, p.t]);
    if (p.kind === "judged" && p.unheard) { out.push(["Another voice left out", "Speech here was held as a possible turn. Its voice checks were closer to the other voice heard in this session than to the user, so Isolation was not run on it and nothing went to the assistant.", p.t]); continue; }
    if (p.kind === "judged" && p.left_out) { out.push(["Another voice left out", `Speech here was held as a possible turn and scored ${p.score} as a whole, under the ${p.line} it needs. Nothing went to the assistant.`, p.t]); continue; }
    if (p.kind === "isolated" && p.left_out) { out.push(["Another voice left out", p.kept_db != null
      ? `Speech here was held as a possible turn. Isolation found none of the user's voice in it and returned it ${Math.abs(p.kept_db)} dB quieter. As a whole it scored ${p.score}, under the ${p.line} it needs. Nothing went to the assistant.`
      : `Speech here was held as a possible turn. What Isolation returned for it was not closer to the user than to the other voice heard in this session by the ${p.line} it needs (${p.score}). Nothing went to the assistant.`, p.t]); continue; }
    if (p.kind === "isolated" && p.failed) out.push(["Isolation not used", `The GPU didn't answer, so this part went to the assistant as Trimming left it, with other voices still under the user's words. Ground Control starts Isolation on the GPU again for the next command.`, p.t]);
    else if (p.kind === "isolated" && p.skipped) out.push(["Isolation not used", `Isolation returned this part ${Math.abs(p.kept_db)} dB quieter, as if the user wasn't talking in it. It went to the assistant as Trimming left it, with other voices still under the user's words. Isolation took ${p.ms} ms.`, p.t]);
    else if (p.kind === "isolated") out.push(["Isolation", `Other voices under the user's words were removed in ${p.ms} ms. The command went to the assistant after that.`, p.t]);
  }
  return out.sort((a, b) => a[2] - b[2]).map(([title, desc, t]) => h("div", { class: "row" },
    h("div", { class: "grow" }, h("div", { class: "rt" }, title), h("div", { class: "rd" }, desc)),
    h("span", { class: "t" }, `${Number(t).toFixed(2)} s`)));
}

/** What the Wake Word Aligner found at the start of the stream and what was done with it. */
function wakeWordStep(p, word) {
  const name = word ? `"${word}"` : "the wake word";
  const title = "Wake Word Aligner";
  if (!p.found) return [title, `The end of ${name} wasn't clear enough to tell from the command, so nothing was left out.`, 0];
  if (!p.t) return [title, `${name[0].toUpperCase()}${name.slice(1)} was over before the command's audio started. Nothing to leave out.`, 0];
  const how = p.rule === "pause" ? "The user paused after it." : "The command started right after it.";
  return [title, `The first ${Number(p.t).toFixed(2)} s was still ${name} and was left out. Finding it took ${p.ms} ms. ${how}`, p.t];
}

function player(path, title, desc, length) {
  const audio = new Audio();
  const btn = h("button", { class: "play", "aria-label": `Play ${title.toLowerCase()}` }, icon("play", 20));
  // The spectrogram under the row, filled in once the audio is here, with
  // a line that follows playback.
  const canvas = h("canvas", { role: "img", "aria-label": `Spectrogram of ${title.toLowerCase()}` });
  const head = h("div", { class: "ph" });
  const spec = h("div", { class: "spec", hidden: true }, canvas, head);
  const lv = h("div", { class: "lv", hidden: true });
  const info = { path, title, canvas, spec, lv, at: null };
  const follow = () => {
    if (audio.duration) head.style.left = `${(info.at ? info.at(audio.currentTime) : audio.currentTime / audio.duration) * 100}%`;
    if (!audio.paused) requestAnimationFrame(follow);
  };
  btn.addEventListener("click", async () => {
    if (!audio.src) {
      try {
        // The spectrogram's copy when it is here, else a download.
        audio.src = info.buf ? URL.createObjectURL(new Blob([info.buf], { type: "audio/wav" })) : await audioUrl(path);
      } catch { toast("That audio wasn't kept."); return; }
    }
    if (audio.paused) { audio.play(); btn.replaceChildren(icon("pause", 20)); spec.classList.add("on"); requestAnimationFrame(follow); }
    else { audio.pause(); btn.replaceChildren(icon("play", 20)); }
  });
  audio.addEventListener("ended", () => { btn.replaceChildren(icon("play", 20)); spec.classList.remove("on"); });
  return [h("div", { class: "row" }, btn,
    h("div", { class: "grow" }, h("div", { class: "rt" }, title), h("div", { class: "rd" }, desc)),
    h("span", { class: "t" }, length), spec, lv), audio, info];
}

/** What went to the assistant, put back where it was in what the kiosk heard,
 * with silence for what was left out, when the pieces add up to it. ``lead``:
 * seconds of silence at the start of it that the kiosk never sent. */
function placed(all, released, total, lead = 0) {
  const samples = all.subarray(Math.round(lead * 16000));
  const sum = released.reduce((n, [a, b]) => n + (b - a), 0);
  if (!released.length || Math.abs(sum * 16000 - samples.length) > 800) return all;
  const out = new Int16Array(Math.round(total * 16000));
  let from = 0;
  for (const [a, b] of released) {
    const n = Math.round((b - a) * 16000);
    out.set(samples.subarray(from, from + n).subarray(0, Math.max(0, out.length - Math.round(a * 16000))), Math.round(a * 16000));
    from += n;
  }
  return out;
}

/** Where ``t`` seconds into the sent audio was in what the kiosk heard. */
function heardAt(t, released, lead = 0) {
  t = Math.max(0, t - lead);
  for (const [a, b] of released) {
    if (t <= b - a) return a + t;
    t -= b - a;
  }
  return released.length ? released[released.length - 1][1] : 0;
}

// The level tiles under a spectrogram: what each is, how far its meter goes
// from the bottom of the scale to its top and what it means.
const TILES = [
  ["over", "Speech over noise", "dB", 0, 60, "How far the speech stood over the noise floor: Speech minus Noise floor. The higher it is, the clearer the voice was over the room."],
  ["peak", "Peak", "dBFS", -90, 0, "The loudest single sample. At 0 dBFS the audio clipped: it was louder than the microphone or the kiosk's audio processing can record, which distorts it."],
  ["speech", "Speech", "dBFS", -90, 0, "How loud the voice was: the average of the 10 loudest 20 ms stretches."],
  ["average", "Average", "dBFS", -90, 0, "The level across the whole recording, speech and pauses together. Silence put in for what was left out doesn't count."],
  ["floor", "Noise floor", "dBFS", -90, 0, "How loud it was between words: the room, a TV, a fan. It's the level 20% of the way up from the quietest 20 ms stretches."],
];

// One decimal, with a real minus sign and no minus on a zero.
function num(v) {
  const r = Math.round(v * 10) / 10;
  return r < 0 ? `\u2212${(-r).toFixed(1)}` : r.toFixed(1);
}

const shown = (key, v) => (key === "over" && Math.round(v * 10) > 0 ? `+${num(v)}` : num(v));

/** Every level of one recording, with what each means. */
function levelsDialog(title, l) {
  inform({
    title: `${title} levels`,
    content: h("div", { class: "lvd" },
      TILES.map(([key, label, unit, , , about]) => h("div", { class: `lvdr ${key === "over" ? "key" : ""}` },
        h("div", { class: "lvdh" },
          h("span", { class: "lvdk" }, key === "peak" && l.clipped ? "Peak, clipped" : label),
          h("span", { class: `lvdv ${key === "peak" && l.clipped ? "warn" : ""}` }, shown(key, l[key]), h("small", {}, unit))),
        h("p", { class: "lvdx" }, about))),
      h("p", { class: "lvdn" }, "dBFS counts down from 0, the loudest a recording can be: \u221220 dBFS is quieter than \u221210 dBFS.")),
  });
}

function levelTiles(title, l) {
  const open = () => levelsDialog(title, l);
  return TILES.map(([key, label, unit, lo, hi]) => {
    const v = l[key];
    const warn = key === "peak" && l.clipped;
    return h("button", { class: `lvc ${key === "over" ? "key" : ""} ${warn ? "warn" : ""}`, onclick: open,
      "aria-label": `${label}: ${shown(key, v)} ${unit}. Show all the levels` },
      h("span", { class: "lvk" }, warn ? "Peak, clipped" : label),
      h("span", { class: "lvv" }, shown(key, v), h("small", {}, unit)),
      h("span", { class: "lvm" }, h("i", { style: { width: `${Math.max(0, Math.min(1, (v - lo) / (hi - lo))) * 100}%` } })));
  });
}

/** Each row's spectrogram, all against the loudest sound the kiosk heard, so
 * what was left out or removed shows up dark. */
async function spectrograms(rows, raw) {
  const levels = await Promise.all(rows.map((r) => (r.path.endsWith("/audio/raw") && raw ? raw : audioBytes(r.path))
    .then((buf) => {
      r.buf = buf;
      // The levels of the audio itself, not of the silence it's placed in.
      const l = loudness(pcm(buf));
      if (l) { r.lv.replaceChildren(...levelTiles(r.title, l)); r.lv.hidden = false; }
      return analyze(r.place ? r.place(pcm(buf)) : pcm(buf));
    }).catch(() => null)));
  const heard = levels[rows.findIndex((r) => r.path.endsWith("/audio/raw"))];
  const top = heard ? heard.max : Math.max(...levels.filter(Boolean).map((l) => l.max));
  rows.forEach((r, i) => {
    if (!levels[i] || !Number.isFinite(top)) return;
    draw(r.canvas, levels[i], top);
    r.spec.hidden = false;
  });
}

// How a turn started, when it wasn't a wake word.
// ask_question and start_conversation reach a kiosk the same way. It says
// which one as soon as Home Assistant's pipeline starts. A session that only
// says "question" never got that far and shows neither.
const STARTED = {
  ask_question: "Ask Question", start_conversation: "Start Conversation", follow_up: "Follow-up", manual: "Started by hand",
};

const ENGINES = { vswakeword: "vsWakeWord", microwakeword: "microWakeWord", openwakeword: "openWakeWord", pryon: "Amazon" };

/** What the kiosk said about how the session started: the wake word and what
 * heard it, the assistant and Talk right after the wake word. */
function wakeChips(wake) {
  if (!wake) return [];
  const engine = wake.wake_word_engine ? ENGINES[wake.wake_word_engine.toLowerCase()] || wake.wake_word_engine : "";
  const word = wake.wake_word ? chip(`Wake word: ${wake.wake_word}${engine ? `, ${engine}` : ""}`, "big") : null;
  if (word && wake.wake_word_model) word.title = `Wake word model: ${wake.wake_word_model}`;
  return [
    word,
    wake.assistant ? chip(wake.assistant === "realtime" ? "Realtime assistant" : "Assist pipeline", "big") : null,
    STARTED[wake.interaction] ? chip(STARTED[wake.interaction], "big") : null,
    wake.seamless ? chip("Talk right after the wake word", "big") : null,
  ];
}

export async function render({ id }) {
  const [record, { people }] = await Promise.all([api(`api/sessions/${id}`), api("api/people")]);
  if (record.kind === "vision") return renderVision(record);
  const stats = record.stats || {};
  const speaker = stats.speaker || {};
  const person = people.find((p) => p.id === speaker.id);
  const color = person?.color || "var(--primary)";
  const total = stats.audio_s || 1;
  const where = record.device_name || record.device_id || "Unknown kiosk";
  const end = (record.events || []).find((e) => e.type === "end_of_speech");
  const sent = end && end.reason !== "silence" ? end.t : total;

  const wave = h("div", { class: "wave", role: "img", "aria-label": "Waveform of the session" });
  let audios = [];
  const follow = record.mode === "follow";
  const raw = record.has_audio ? audioBytes(`api/sessions/${id}/audio/raw`) : null;
  if (raw) {
    raw.then((buf) => wave.replaceChildren(...(follow
      ? followWave(pcm(buf), total, record.released || [], color)
      : waveform(pcm(buf), total, record.events || [], color)))).catch(() => {});
  }
  const marker = end && !follow ? h("div", { class: "mk", style: { left: `${(sent / total) * 100}%` } }, h("span", { class: "mklbl" }, `Stopped, ${secs(sent)}`)) : null;

  const who = h("select", { class: "fld", "aria-label": "Person" }, people.map((p) => h("option", { value: p.id, selected: p.id === speaker.id }, p.name)));
  const add = async () => {
    try {
      const r = await api(`api/sessions/${id}/enroll`, { method: "POST", json: { person_id: who.value } });
      toast(`Added ${r.added_s} s of speech to ${r.name}`);
    } catch (err) { toast(err.message); }
  };
  const remove = async () => {
    if (!await ask({ title: "Delete this session?", text: "Its audio is deleted too.", ok: "Delete", danger: true })) return;
    await api(`api/sessions/${id}`, { method: "DELETE" });
    location.hash = "#/sessions";
  };
  // Everything about this session in one zip, for an issue report.
  const download = async () => {
    try {
      const a = h("a", { href: await audioUrl(`api/sessions/${id}/download`), download: `ground-control-session-${id}.zip` });
      a.click();
    } catch (err) { toast(`Couldn't download the session: ${err.message}`); }
  };

  let audioSection = [h("div", { class: "empty" }, "Audio keeping was off for this session.")];
  let specs = [];
  if (record.has_audio) {
    const [rawRow, rawAudio, rawSpec] = player(`api/sessions/${id}/audio/raw`, "As heard", "As heard by the kiosk's microphone.", secs(total));
    const lead = record.lead_in_s ? ` ${record.lead_in_s} s of silence was put in front of it because the command started right after the wake word.` : "";
    const [cleanRow, cleanAudio, cleanSpec] = follow
      ? player(`api/sessions/${id}/audio/clean`, "Sent to the assistant",
        (stats.isolation_skipped || stats.isolation_failed ? "Only the user's speech: other voices left out between their words. Isolation wasn't used for all of it, see Steps."
          : stats.isolated_ms !== undefined ? "Only the user's speech: other voices left out between their words and removed under them."
          : "Only the user's speech without the other voices.") + lead,
        secs((stats.released_s ?? 0) + (record.lead_in_s || 0)))
      : player(`api/sessions/${id}/audio/clean`, record.mode === "gated" ? "Sent to the assistant" : "Cleaned",
        "Other voices replaced with silence.", secs(sent));
    audioSection = [rawRow, cleanRow];
    audios = [rawAudio, cleanAudio];
    // In follow mode, the sent audio lines up with what was heard.
    if (follow) {
      cleanSpec.place = (samples) => {
        const lead = record.lead_in_s || 0;
        const out = placed(samples, record.released || [], total, lead);
        // The line follows the sent audio on the same timeline.
        if (out !== samples) cleanSpec.at = (t) => heardAt(t, record.released, lead) / total;
        return out;
      };
    }
    specs = [rawSpec, cleanSpec];
  }
  if (record.wake_s) {
    // Kept apart from the rest: the kiosk heard it before the session began.
    const [wakeRow, wakeAudio, wakeSpec] = player(`api/sessions/${id}/audio/wake`, "Wake word",
      "The wake word as the kiosk recorded it. Whoever says it is the user for this command. It's also the first sample of their voice.", secs(record.wake_s));
    audioSection = [wakeRow, ...(record.has_audio ? audioSection : [])];
    audios = [wakeAudio, ...audios];
    specs = [wakeSpec, ...specs];
  }
  spectrograms(specs, raw?.catch(() => null));

  return {
    nodes: [
      shell("sessions", {
        back: "#/sessions", title: `${where}, ${clock(record.started)}`,
        actions: [
          h("button", { class: "ib", "aria-label": "Download the session", title: "Download the session: its audio, details and log", onclick: download }, icon("download")),
          h("button", { class: "ib", "aria-label": "Delete session", onclick: remove }, icon("trash")),
        ],
      }),
      h("main", { class: "page" },
        h("div", { class: "chips" },
          speaker.name ? chip(speaker.score != null ? `${speaker.name}, ${speaker.score}` : speaker.name, "big", h("span", { class: "dot", style: { background: color, width: "10px", height: "10px" } }))
            : chip(`${speaker.by === "answer" ? "Short answer" : "Unknown voice"}${speaker.score ? `, ${speaker.score}` : ""}`, "big"),
          outcomeChip(stats),
          chip(`${modeLabel(record.mode)} mode`, "big"),
          chip(`Answers: ${answersLabel(record, people)}`, "big"),
          chip(`${record.model === "light" ? "Light" : "Accurate"} model`, "big"),
          wakeChips(record.wake)),
        section("Timeline"),
        h("div", { class: "card tl" },
          follow
            ? h("div", { class: "legend" },
              h("span", {}, h("i", { class: "swb", style: { background: color } }), `${speaker.name ? `${speaker.name}'s` : "The user's"} speech sent to the assistant`),
              h("span", {}, h("i", { class: "swb", style: { background: "var(--other)", opacity: 0.45 } }), "Left out"))
            : h("div", { class: "legend" },
              h("span", {}, h("i", { class: "swb", style: { background: color, opacity: 0.55 } }), "Before speech"),
              h("span", {}, h("i", { class: "swb", style: { background: color } }), speaker.name || "Speaker"),
              h("span", {}, h("i", { class: "swb", style: { background: "var(--other)" } }), "Other voice, left out"),
              h("span", {}, h("i", { class: "swb", style: { background: "var(--other)", opacity: 0.3 } }), "After the end")),
          h("div", { style: { position: "relative", marginLeft: "30px" } }, wave, marker,
            (record.followups || []).map((t) => h("div", { class: "mk", style: { left: `${(t / total) * 100}%` } },
              h("span", { class: "mklbl" }, "Follow-up")))),
          chart(record, total, color),
          h("div", { class: "axis" }, h("span", {}, "0 s"), h("span", {}, secs(total / 2)), h("span", {}, secs(total))),
          chartKey(record, color)),
        h("div", { class: "cols" },
          h("section", {}, section("Steps"), h("div", { class: "card" }, steps(record),
            h("div", { class: "row" }, h("div", { class: "grow" }, h("div", { class: "rt" }, "Result"), h("div", { class: "rd" }, `${outcome(stats)}. ${stats.checks ?? 0} checks used ${stats.cpu_ms ?? 0} ms of CPU.${stats.isolated_ms !== undefined ? ` Isolation: ${stats.isolated_ms} ms.` : ""}`)))),
            section("Use this voice"),
            h("div", { class: "card" },
              h("div", { class: "row" },
                h("div", { class: "grow" }, h("div", { class: "rt" }, "Add to a person"), h("div", { class: "rd" }, follow ? "Only the user's speech is added. The voices that were left out are not." : "Only the speech before another voice is added.")),
                who, h("button", { class: "btn out", onclick: add, disabled: !record.has_audio || !people.length }, "Add")))),
          h("section", {}, section("Audio"), h("div", { class: "card" }, audioSection)))),
    ],
    cleanup: () => audios.forEach((a) => { a.pause(); if (a.src) URL.revokeObjectURL(a.src); }),
  };
}
