# Kiosk Satellite Ground Control

Ground Control is a companion service for [Kiosk Satellite](https://github.com/jxlarrea/kiosk-satellite). It runs on a computer or server in your home and handles the voice and vision processing that kiosks don't have the computing power for.

Kiosk Satellite Ground Control currently has two components: **Voice Match** which does speaker voice recognition, trimming, wake word alignment and isolation. **Vision Match** which does facial recognition (coming soon).

**Voice Match** works with Home Assistant's Assist and the realtime assistants in Kiosk Satellite:

- **Recognition:** who is speaking to a Kiosk Satellite, from the voiceprints of the people enrolled.
- **Trimming:** other voices, like a TV, are cut out between the user's words, so a TV in the background doesn't keep the assistant listening.
- **Wake Word Aligner:** the end of the wake word is left out of the command when a kiosk has **Talk right after the wake word** on.
- **Isolation:** other voices talking over the user's words are removed too. Best on an NVIDIA GPU.

## The panel

Ground Control adds a panel to Home Assistant's sidebar. Enroll people there, see each kiosk live, go through past sessions, change the settings and read the logs. Changes made in the panel are kept over the options below, which are where things start.

## What Kiosk Satellite sends

With **Enable Trimming** on in Kiosk Satellite, a kiosk sends everything its microphone hears, and Home Assistant or the realtime assistant gets back only the speech of the person who said the wake word. A TV or another person before, between and after their words is cut out, and the command ends when they stop. The assistant hears them about a second late while they talk, the time it takes to tell voices apart.

With **Talk right after the wake word** on in Kiosk Satellite too, the end of the wake word is left out of the command, like the "xa" of "Alexa". **Use Wake Word Aligner** under **Settings > Voice Match** turns that off.

With **Isolation** on in the panel, under **Settings > Voice Match**, the command goes out when the user stops talking, with the voices under their words removed too. Each kiosk can turn it off with **Enable Isolation** in Kiosk Satellite. It needs the Accurate speaker model. On a CPU each command waits about a second for it, so it is meant for an NVIDIA GPU.

## Options

- **model:** the speaker model.
  - `auto` picks Accurate on an NVIDIA GPU, an Apple M-series chip or an x86 CPU with more than 4 cores, and Light on other ARM CPUs and smaller x86 CPUs.
  - `accurate` makes the fewest mix-ups between people and with TV voices. Use it on a desktop-class CPU or a GPU.
  - `light` uses a fraction of the CPU, for a Raspberry Pi, Home Assistant Green or an Intel N100. A TV voice cutting in ends the command about a second later.

  Changing the model rebuilds everyone's voiceprint from the enrollment audio. Nobody has to record again.
- **threads:** threads per check. `0` uses half the CPUs, up to 4.
- **keep_days:** how many days saved sessions are kept, for the Sessions page.
- **keep_audio:** keep each session's audio, so it can be played back or added to a person.
- **mdns_address:** optional. Kiosks find Ground Control at this machine's address on its main network, the one with the default route. Set an address only when kiosks can't reach that one, like kiosks on another VLAN this machine also has an address on. Every kiosk then gets that address instead.

## Files and backups

People, voiceprints, kiosks and saved sessions are in the app's folder, `addon_configs/…_kiosk_satellite_ground_control`, which the Samba share and File editor apps show. A Home Assistant backup that includes this app keeps all of it, so a restore brings back everyone's voiceprint.

## Reporting an issue

Open **Logs** in the panel and select **Download**. The file holds what Ground Control did since it started, with the version, the speaker model, the settings and every kiosk at the top. Attach it to the issue. Tokens never reach the log. If you can make the problem happen again, turn on **Detailed logs** first, repeat it and then download. Detailed logs stay on until Ground Control restarts.

## Kiosk Satellite

Kiosks sign in with the Home Assistant token they already have. Kiosks on the same network find the service on their own. For a kiosk on another network, enter this machine's address and port 10900 under **Settings > Ground Control** in Kiosk Satellite.

Ground Control only accepts kiosks running Kiosk Satellite. A kiosk that connects for the first time waits until you approve it on the panel's Kiosks page. Its page there also shows which Voice Match features it has on.

## More

The [documentation](https://github.com/jxlarrea/kiosk-satellite-ground-control/blob/main/docs/getting-started.md) covers kiosks, people, Voice Match, sessions and every setting.
