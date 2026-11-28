// Settings: one page per area, picked from a list on the left like People and
// Kiosks. Inside a page every group is the same: a section title with a line
// under it, then a card of rows.
import { api, loggedIn, logOut } from "../api.js";
import { shell } from "../app.js";
import { ask, chip, h, icon, pageHead, section, toast, toggle } from "../ui.js";

const TEMPLATE = [
  "{%- if llm_context.device_id %}",
  "{%- set speaker_entity = device_entities(llm_context.device_id) | select('match', 'sensor\\..*_speaker$') | first | default('') %}",
  "{%- if speaker_entity %}",
  "The person speaking is in the \"{{ state_attr(speaker_entity, 'friendly_name') }}\" sensor. It changes from one request to the next, so whenever the answer depends on who is speaking (their name, \"my\", \"me\", \"mine\"), call GetLiveContext first and read that sensor. Never reuse a name from earlier in the conversation. If the sensor is empty, unknown or unavailable, nobody was identified.",
  "{%- endif %}",
  "{%- endif %}",
].join("\n");

const AREAS = [
  ["voice-match", "Voice Match", "mark", "Recognition, Trimming, Wake Word Aligner and Isolation"],
  ["vision-match", "Vision Match", "eye", "Face Detection, Hand Gestures and Face Recognition"],
  ["general", "General", "tune", "Engine, sessions, agents and kiosks"],
];

async function copy(text) {
  try { await navigator.clipboard.writeText(text); }
  catch {
    // Inside the Home Assistant panel's frame the clipboard API can be off.
    const area = h("textarea", { style: { position: "fixed", opacity: "0" } });
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  toast("Copied");
}

function row(title, desc, control, id) {
  return h("div", { class: "row" },
    h("div", { class: "grow" }, h("div", { class: "rt" }, id ? h("label", { for: id }, title) : title), h("div", { class: "rd" }, desc)),
    control);
}

const hint = (text) => h("div", { class: "hint" }, icon("info", 18), text);

/** A Light or Accurate choice, the same for the speaker model and each vision
 * model. ``offered`` lists the keys this install can use. */
function modelSeg(name, current, offered, onPick) {
  const seg = h("div", { class: "seg", role: "radiogroup", "aria-label": name });
  for (const [key, label] of [["light", "Light"], ["accurate", "Accurate"]]) {
    const on = current === key;
    const b = h("button", { class: on ? "on" : "", role: "radio", "aria-checked": String(on), disabled: !offered.includes(key) }, label);
    b.addEventListener("click", async () => {
      if (on || !offered.includes(key)) return;
      b.setAttribute("aria-busy", "true");
      try {
        await onPick(key, label);
      } catch (err) {
        toast(err.message);
        b.removeAttribute("aria-busy");
      }
    });
    seg.append(b);
  }
  return seg;
}

// The line under a Runs on row. ``what``: "Isolation runs" or "Each model runs".
const runsOnText = (s, what) => (s.vulkan ? `${what} on the GPU when that is faster than this machine's CPU.`
  : s.openvino ? (what.startsWith("Isolation") ? "Isolation runs on the CPU in the OpenVINO image." : `Vision Match runs on ${s.openvino} in the OpenVINO image.`)
  : "Set by the CUDA option in the same way as the speaker model.");

const defaultText = (key) => ` The default here is ${key === "accurate" ? "Accurate" : "Light"}.`;

const size = (bytes) => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} KB`);

function areas(current) {
  return h("div", { class: "card" },
    AREAS.map(([key, label, ico, desc]) => h("a", {
      class: `row nowrap ${key === current ? "sel" : ""}`, href: `#/settings/${key}`, "aria-current": key === current ? "page" : null,
    }, h("span", { class: "ico" }, icon(ico, 24)),
    h("div", { class: "grow" }, h("div", { class: "rt" }, label), h("div", { class: "rd" }, desc)))));
}

export async function render({ area } = {}) {
  const current = AREAS.some(([key]) => key === area) ? area : "voice-match";
  const s = await api("api/settings");
  const runsOn = (what) => runsOnText(s, what);
  let poll = null;

  const save = async (body) => {
    try { await api("api/settings", { method: "PATCH", json: body }); toast("Saved"); }
    catch (err) { toast(err.message); }
  };
  const number = (id, value, key) => {
    const input = h("input", { class: "fld num", id, value: String(value), inputmode: "decimal" });
    input.addEventListener("change", () => save({ [key]: Number(input.value) }));
    return input;
  };

  // ── Voice Match ──────────────────────────────────────────────────────────

  const voiceMatch = () => {
    const progress = h("div", { class: "row", role: "status", style: { display: s.rebuilding ? "" : "none" } },
      h("div", { class: "grow" }, h("div", { class: "rt" }, "Updating voiceprints"),
        h("div", { class: "rd" }, "Kiosks keep using the current model until this finishes.")));
    const speaker = modelSeg("Speaker model", s.model, Object.keys(s.models), async (key, label) => {
      const r = await api("api/settings", { method: "PATCH", json: { model: key } });
      if (!r.rebuilding) {
        toast(`Now using ${label}`);
        location.reload();
        return;
      }
      progress.style.display = "";
      clearInterval(poll);
      poll = setInterval(async () => {
        const now = await api("api/settings");
        if (!now.rebuilding) {
          clearInterval(poll);
          toast(`Now using ${now.models[now.model]}`);
          location.reload();
        }
      }, 1000);
    });

    const usable = s.isolator_available && s.model === s.isolator_model;
    const where = h("span", {});
    const showWhere = (onGpu) => {
      where.replaceChildren(onGpu === null || onGpu === undefined ? h("span", { class: "val" }, "Off")
        : chip(onGpu ? (s.vulkan ? "GPU (Vulkan)" : "NVIDIA GPU (CUDA)") : "CPU", onGpu ? "c-ok" : ""));
    };
    showWhere(s.isolator ? s.isolator_on_gpu : null);
    const isolation = toggle(s.isolator && usable, "Use Isolation", async (v) => {
      await save({ isolator: v });
      const now = await api("api/settings");
      showWhere(now.isolator ? now.isolator_on_gpu : null);
    });
    if (!usable) isolation.disabled = true;
    const isolationNote = !s.isolator_available ? "The isolation model is not in this image."
      : s.model !== s.isolator_model ? "Isolation needs the Accurate speaker model."
      : s.vulkan_slow ? `Isolation runs on the CPU: ${s.vulkan} was slower than this machine's CPU when Ground Control timed both.`
      : s.vulkan ? `Each command waits for Isolation once the user stops talking. It runs on ${s.vulkan}.`
      : "Each command waits for Isolation once the user stops talking: under a tenth of a second on an NVIDIA GPU and about half a second on a fast x86 CPU. It is too slow on a Raspberry Pi.";

    const aligner = toggle(s.aligner && !s.aligner_missing, "Use Wake Word Aligner", (v) => save({ aligner: v }));
    if (s.aligner_missing) aligner.disabled = true;
    const alignerNote = s.aligner_missing === "model" ? "The Wake Word Aligner model is not in this install."
      : s.aligner_missing ? "The Wake Word Aligner needs espeak-ng, which is not installed. The Docker images and the Home Assistant app have it."
      : "It works with any wake word and any wake word engine. It doesn't delay the command.";

    return [
      section("Recognition", "Settings for matching voices to the voiceprints of the people enrolled."),
      h("div", { class: "card" },
        row("Speaker model", `Accurate makes the fewest mix-ups between people and with other voices. Light uses about a third of the CPU and suits a Raspberry Pi or Home Assistant Green. With Light, the turn ends about a second later when another voice cuts in. Voiceprints are rebuilt from the enrollment audio when this changes, so nobody has to record again.${defaultText(s.default_model)}`, speaker),
        progress,
        row("Threshold", `How close a voice has to be to count as someone enrolled. Higher means fewer mix-ups and more unknown voices. Each model has its own threshold. The default is ${s.default_threshold}.`,
          number("thr", s.threshold, "threshold"), "thr"),
        row("Take short answers", "Accepts a short reply to a question from the assistant, like \"yes\" or \"no\", when the voice vaguely sounds like one of the people the kiosk accepts. A reply this short isn't enough to recognize who is speaking, so with this off it is ignored. Only applies to kiosks that answer chosen people.",
          toggle(s.short_answers, "Take short answers", (v) => save({ short_answers: v })))),

      section("Trimming", "Other voices are cut out between the user's words and long pauses are shortened."),
      h("div", { class: "card" },
        row("Other voice threshold", `Two checks in a row under this score mean someone else is talking. ${s.default_other_threshold} by default.`,
          number("oth", s.other_threshold, "other_threshold"), "oth"),
        hint("Turn Trimming on or off for each kiosk in Kiosk Satellite under Ground Control > Voice Match.")),

      section("Wake Word Aligner", "The end of the wake word, such as the \"xa\" of \"Alexa\", is left out of the command."),
      h("div", { class: "card" },
        row("Use Wake Word Aligner", "For kiosks with Trimming and Talk right after the wake word on in Kiosk Satellite.", aligner),
        hint(alignerNote)),

      section("Isolation", "Uses the user's voiceprint to also remove other voices talking over their words."),
      h("div", { class: "card" },
        row("Use Isolation", "For kiosks with Trimming and Isolation on in Kiosk Satellite.", isolation),
        row("Runs on", runsOn("Isolation runs"), where),
        hint(isolationNote)),
    ];
  };

  // ── Vision Match ─────────────────────────────────────────────────────────

  const visionMatch = () => {
    const names = { face: "Face Detection", hands: "Hand Gestures", recognition: "Face Recognition" };
    const pick = (feature) => modelSeg(`${names[feature]} model`, s.vision_models[feature],
      Object.keys(s.vision_choices[feature] || {}), async (key, label) => {
        await api("api/settings", { method: "PATCH", json: { vision_models: { [feature]: key } } });
        toast(`${names[feature]} now uses ${label}`);
        location.reload();
      });
    const measured = (feature, what) => {
      const ms = s.vision_ms && s.vision_ms[feature];
      return ms == null ? "" : ` On this machine it takes ${ms < 10 ? ms : Math.round(ms)} ms ${what}.`;
    };
    const byDefault = (feature) => defaultText(s.vision_defaults[feature]);
    const where = !s.vision_loaded ? h("span", { class: "val" }, "Loads with the first look")
      : s.vulkan ? chip(s.vision_on_gpu ? "GPU (Vulkan)" : s.vision_gpu_models && s.vision_gpu_models.length ? "GPU (Vulkan) and CPU" : "CPU",
        s.vision_gpu_models && s.vision_gpu_models.length ? "c-ok" : "")
      : chip(s.vision_on_gpu ? (s.openvino ? "Intel GPU (OpenVINO)" : "NVIDIA GPU (CUDA)") : "CPU", s.vision_on_gpu ? "c-ok" : "");
    const looks = s.vision_max_looks
      ? h("span", { class: "val" }, `${s.vision_max_looks.face} face, ${s.vision_max_looks.hands} hand`)
      : h("span", { class: "val" }, "Measured when they load");
    const saved = h("span", { class: "val" }, "");
    const removeAll = h("button", { class: "btn out sm danger", disabled: true }, "Delete all");
    const showFrames = (f) => {
      saved.textContent = f.frames ? `${f.frames} frame${f.frames === 1 ? "" : "s"}, ${size(f.bytes)}` : "None saved";
      removeAll.disabled = !f.frames;
    };
    api("api/frames").then(showFrames).catch(() => { saved.textContent = ""; });
    removeAll.addEventListener("click", async () => {
      if (!await ask({ title: "Delete all frames?", text: "Every frame kept with a vision session is deleted. The sessions keep their timeline, steps and times.", ok: "Delete", danger: true })) return;
      try {
        const gone = await api("api/frames", { method: "DELETE" });
        toast(`Deleted ${gone.frames} frame${gone.frames === 1 ? "" : "s"}`);
        showFrames({ frames: 0, bytes: 0 });
      } catch (err) { toast(err.message); }
    });
    return [
      section("Models", "Choose Light or Accurate for each feature. Kiosks with Vision Match on use these models."),
      h("div", { class: "card" },
        row("Face Detection", `Both models found every face looking at the kiosk on the test set and let the same turned heads through. Accurate takes about 3 times as long.${byDefault("face")}${measured("face", "a look")}`, pick("face")),
        row("Hand Gestures", `Accurate reads the finger count right more often at 3 m and in dim light. The hand model runs on every look, so on a CPU Accurate lowers the looks a second each kiosk gets.${byDefault("hands")}${measured("hands", "a look")}`, pick("hands")),
        row("Face Recognition", `Accurate recognizes small faces in dim light more often. A recognized face is checked again only every 2 seconds, so Accurate costs little even on a CPU. Faces are rebuilt from their pictures, so nobody has to enroll again.${byDefault("recognition")}${measured("recognition", "a face")}`, pick("recognition")),
        row("Runs on", runsOn("Each model runs"), where),
        row("Looks a second", "For one kiosk, measured on this machine when the models load. Kiosks look at most 2 times a second for faces and 4 for hands.", looks),
        s.vision_error ? hint(s.vision_error)
          : hint("Turn Vision Match on for each kiosk in Kiosk Satellite under Ground Control > Vision Match.")),

      section("Hand Gestures", "Settings for reading raised hands on kiosk frames."),
      h("div", { class: "card" },
        row("Hand sensitivity", `From 1 to 100. With a higher value, smaller hands farther away count too. With a lower value, a hand that is only a small part of the frame, such as one cut off at its edge, counts as no hand. At ${s.hand_sensitivity} the smallest hand counted is ${Math.round(s.hand_min_size * 100)}% of the frame. The default is 50.`,
          number("hsens", s.hand_sensitivity, "hand_sensitivity"), "hsens")),

      section("Frames", "Camera frames are kept with each vision session and shown on its page in Sessions."),
      h("div", { class: "card" },
        row("Keep frames", `Up to 8 frames a session: the first look, a face looking at the kiosk, each new finger reading and the last look. They are deleted with the session after ${s.keep_days} day${s.keep_days === 1 ? "" : "s"}.`,
          toggle(s.keep_frames, "Keep frames", (v) => save({ keep_frames: v }))),
        row("Delete all frames", "The sessions stay but their frames are deleted.", h("span", { class: "cell" }, saved, removeAll))),

      section("Face Recognition", "Settings for matching faces to the faces enrolled on the People page."),
      h("div", { class: "card" },
        row("Threshold", `How close a face has to be to count as someone enrolled. Higher means fewer mix-ups and more unknown faces. Each Face Recognition model has its own threshold. The default is ${s.default_face_threshold}.`,
          number("fthr", s.face_threshold, "face_threshold"), "fthr"),
        row("Faces enrolled", "How many people on the People page have a face enrolled.",
          h("a", { class: "val", href: "#/people" }, String(s.faces_enrolled))),
        hint("A name needs a face looking at the kiosk on two looks in a row or one very close match. Turn Face Recognition on for each kiosk in Kiosk Satellite under Ground Control > Vision Match.")),
    ];
  };

  // ── General ──────────────────────────────────────────────────────────────

  const host = `${location.hostname}:${s.port}`;
  const general = () => [
    section("Engine", "Where voices are checked."),
    h("div", { class: "card" },
      row("Runs on", s.cuda && !s.on_gpu ? "CUDA is on but no GPU is usable, so the CPU is used."
        : s.openvino ? `Voices are checked on the CPU in the OpenVINO image. Vision Match runs on ${s.openvino}.`
        : s.vulkan ? `In the Vulkan image each model runs on ${s.vulkan} when that is faster than this machine's CPU. Ground Control times both when a model loads.`
        : "Set by the CUDA option.",
        chip(s.on_gpu ? (s.vulkan ? "GPU (Vulkan)" : "NVIDIA GPU (CUDA)") : "CPU", s.on_gpu ? "c-ok" : "")),
      row("Threads", "Threads per check. Set in the app's options.", h("span", { class: "val" }, String(s.threads))),
      hint("CUDA is available in the Docker image on a machine with an NVIDIA GPU.")),

    section("Sessions", "The sessions shown on the Sessions page."),
    h("div", { class: "card" },
      row("Keep sessions for", "Older sessions, their audio and their frames are deleted.",
        (() => {
          const sel = h("select", { class: "fld", id: "keep" },
            [1, 7, 30, 90].map((d) => h("option", { value: d, selected: d === s.keep_days }, `${d} day${d === 1 ? "" : "s"}`)));
          sel.addEventListener("change", () => save({ keep_days: Number(sel.value) }));
          return sel;
        })(), "keep"),
      row("Keep audio", "Needed to replay a session or add it to a person.",
        toggle(s.keep_audio, "Keep audio", (v) => save({ keep_audio: v })))),

    section("Conversation agents", "Tells an LLM agent in Home Assistant who is speaking. One copy works for every kiosk."),
    h("div", { class: "card" },
      row("Speaker in the agent's instructions", "Paste at the end of the agent's instructions in Home Assistant.",
        h("button", { class: "ib", "aria-label": "Copy the template", onclick: () => copy(TEMPLATE) }, icon("copy"))),
      h("pre", { class: "code" }, TEMPLATE),
      hint("The agent needs the Assist API and each kiosk's Speaker sensor exposed to Assist.")),

    section("Kiosks", "Kiosks on the same network connect automatically with their Home Assistant token, so there is nothing to enter."),
    h("div", { class: "card" },
      row("Address", "For a kiosk on another network, enter this in Kiosk Satellite.",
        h("span", { class: "cell" }, h("span", { class: "val" }, host),
          h("button", { class: "ib", "aria-label": "Copy the address", onclick: () => copy(host) }, icon("copy", 20))))),

    // Standalone only: the browser logged in with Home Assistant.
    loggedIn() ? section("This browser") : null,
    loggedIn() ? h("div", { class: "card" },
      row("Signed in with Home Assistant", "Signing out asks Home Assistant to end this sign-in.",
        h("button", { class: "btn out sm", onclick: logOut }, "Sign out"))) : null,
  ];

  const content = { "voice-match": voiceMatch, "vision-match": visionMatch, general }[current]();
  return {
    nodes: [
      shell("settings"),
      h("main", { class: "page" },
        pageHead("Settings"),
        h("div", { class: "split" },
          h("section", { class: "left", "aria-label": "Settings" }, areas(current)),
          h("section", { class: "right", "aria-label": AREAS.find(([key]) => key === current)[1] }, content))),
    ],
    cleanup: () => clearInterval(poll),
  };
}
