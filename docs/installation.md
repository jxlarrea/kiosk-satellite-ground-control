# Installation

Kiosk Satellite Ground Control runs on a CPU or GPU accelerated (CUDA, OpenVINO and Vulkan) as a Home Assistant app or a Docker container. Kiosks running [Kiosk Satellite](https://github.com/jxlarrea/kiosk-satellite) connect to it over the network.

## Choosing a Machine

Every Voice Match and Vision Match feature runs on the Ground Control machine instead of the kiosks. How long each one takes depends on that machine:

| Machine | Speaker model | Isolation | Vision Match look |
| --- | --- | --- | --- |
| NVIDIA GPU | Accurate | Under a tenth of a second per command | 17 to 34 ms |
| x86 CPU with more than 4 cores | Accurate | About half a second per command | 64 ms on an i9-13900H |
| Mac with an Apple M-series chip | Accurate | Under half a second per command on an M5 | 70 ms on an M5 |
| Intel N100 and other small x86 CPUs | Light | About 3 seconds per command with Accurate | 292 ms |
| Intel GPU with the OpenVINO image | As on the machine's CPU | As on the machine's CPU | 81 ms on an Intel N100, 25 ms on an Iris Xe |
| AMD GPU with the Vulkan image | As on the machine's CPU | Depends on the GPU | Depends on the GPU. See [AMD GPU](#amd-gpu) |
| Raspberry Pi 4, Home Assistant Green and other ARM boards | Light | Too slow | 671 ms on a Pi 4 |

The Vision Match column is a look with Face Detection, Face Recognition and Hand Gestures on the machine's default vision models. See [Performance Benchmarks](../README.md#performance-benchmarks) for the measured times. Isolation is optional, so a small machine still runs Recognition, Trimming and the Wake Word Aligner. Vision Match runs on every machine in the table. On a Pi 4, Face Detection dismisses the screensaver about a second after a face looks at the kiosk. See [Vision Models](vision-match.md#vision-models).

On amd64, the CUDA image works with NVIDIA GPUs from the GTX 10 series (Pascal) to the RTX 50 series (Blackwell). On arm64, it works with the GB10 and other Blackwell boards. It needs 1 GB of free GPU memory or 2 GB with Isolation. Vision Match needs about 0.5 GB more.

## Home Assistant App

[![Open your Home Assistant instance and show the add app repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fjxlarrea%2Fkiosk-satellite-ground-control)

Select the button above to add the repository to your Home Assistant or add it by hand:

1. In Home Assistant, go to **Settings > Apps > App store**, open the menu and choose **Repositories**.
2. Add `https://github.com/jxlarrea/kiosk-satellite-ground-control`.
3. Install **Kiosk Satellite Ground Control** and start it.
4. Open **Ground Control** in the sidebar.

The app runs on the CPU. Home Assistant OS has no NVIDIA driver, so use the Docker image on another machine for a GPU.

The app uses the host network, so it is discovered automatically on the same network. It signs in to Home Assistant through the Supervisor, so there is no address or token to enter.

### App Options

These options are on the app's **Configuration** tab:

| Option | Default | What it does |
| --- | --- | --- |
| Speaker model | auto | `auto` picks Accurate on an x86 CPU with more than 4 cores or an Apple M-series chip. It picks Light on other ARM CPUs and smaller CPUs. Use `accurate` or `light` to set it. See [Speaker Models](voice-match.md#speaker-models). |
| Threads | 0 | Threads per check. `0` uses half the CPUs with a maximum of 4. |
| Days to keep sessions | 7 | How long saved sessions stay on the Sessions page. Set it from 1 to 90 days. |
| Keep session audio | on | Keeps each session's audio, so it can be played back or added to a person. |
| mDNS address | empty | Optional. See [Kiosks on Another Network](kiosks.md#kiosks-on-another-network). |

You can also change the speaker model and session keeping in the app's [Settings](settings.md). Those settings then take precedence over these options. The Vision Match models have no option here. Choose them in the app under **Settings > Vision Match**.

### Files and Backups

People, voiceprints, faces, kiosks and saved sessions are stored in the app's own folder at `addon_configs/…_kiosk_satellite_ground_control`. You can open it with the Samba share and File editor apps. A Home Assistant backup that includes the app keeps all of it, so a restore brings back everyone's voiceprint and face. See [Backup and Moving](backup.md) to back it up or move it to another machine.

## Docker

There is an image for each kind of machine:

| Image | Tag | Runs |
| --- | --- | --- |
| [CPU](#cpu) | `latest` | Everything on the CPU of an amd64 or arm64 machine. |
| [NVIDIA GPU (CUDA)](#nvidia-gpu) | `cuda` | Everything on an NVIDIA GPU. |
| [Intel GPU (OpenVINO)](#intel-gpu) | `openvino` | Vision Match on an Intel GPU and Voice Match on the CPU. |
| [AMD GPU (Vulkan)](#amd-gpu) | `vulkan` | Each model on an AMD GPU when it is faster there than on the CPU. |

Docker Compose files are in the repository: [compose.yaml](../compose.yaml) for the CPU, [compose.cuda.yaml](../compose.cuda.yaml) for an NVIDIA GPU, [compose.openvino.yaml](../compose.openvino.yaml) for an Intel GPU and [compose.vulkan.yaml](../compose.vulkan.yaml) for an AMD GPU.

1. Download the file for your machine.
2. Set `HA_URL` to your Home Assistant.
3. Run `docker compose up -d`. For a GPU, name the file, as in `docker compose -f compose.cuda.yaml up -d`.
4. Open `http://THIS_MACHINE:10900` and choose **Log in with Home Assistant**.

All four files use the host network, so Ground Control is discovered automatically on the same network. If you don't use the host network, publish port 10900 and enter the machine's address in each kiosk.

`HA_URL` is used in two ways. Kiosk tokens are checked there and you log in to the web app there from your browser. Use an address your browser can open. Avoid one that only works inside Docker.

The sections below have the command to run each image without Compose and what each one needs.

### CPU

```bash
docker run -d --name ground-control --network host \
  -e HA_URL=http://homeassistant.local:8123 \
  -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control
```

The CPU image runs on any amd64 or arm64 machine. See [Choosing a Machine](#choosing-a-machine) for what each kind of CPU keeps up with.

### NVIDIA GPU

Add `--gpus all` and use the `cuda` tag:

```bash
docker run -d --name ground-control --network host --gpus all \
  -e HA_URL=http://homeassistant.local:8123 \
  -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control:cuda
```

The CUDA image needs:

- An NVIDIA GPU with 1 GB of free memory or 2 GB with Isolation, and about 0.5 GB more for Vision Match. On amd64, it can be any GPU from the GTX 10 series (Pascal) to the RTX 50 series (Blackwell). On arm64, it can be the GB10 or another Blackwell board.
- An NVIDIA driver from the 570 series or newer on amd64 or 580 or newer on arm64.
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

The image is built for amd64 and arm64, so it also runs on an NVIDIA GB10. The CUDA image runs on the GPU or not at all. If it can't use the GPU, it stops and logs a line that starts with `CUDA can't start`. The line gives the reason. For example, no GPU is visible to the container or the driver is too old. Use the CPU image on a machine without a supported GPU.

> **Info:** If Docker runs inside an LXC container on Proxmox, `nvidia-smi` fails with "Failed to initialize NVML: Unknown Error" until you set `nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place` and pass the devices to the container: `--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools`.

#### GPU Memory

This table shows how much GPU memory the CUDA image uses with the Accurate speaker model. It was measured on an RTX 4000 SFF Ada and a GB10:

| | amd64 image (CUDA 12) | arm64 image (CUDA 13) |
| --- | --- | --- |
| Kept, Recognition and Trimming | 0.6 GB | 0.7 GB |
| Kept, with Isolation | 0.8 GB | 0.9 GB |
| Most during a command | 0.9 GB | 0.9 GB |
| Most during a command, with Isolation | 1.6 GB | 1.7 GB |

Ground Control keeps the first two amounts for as long as it runs. While a command runs, it needs up to 0.3 GB more. With Isolation, a command of 10 seconds or longer needs up to 0.8 GB more. That memory is freed once the command is done. Commands that end at the same moment are cleaned one after the other, so the total never goes above that. If the GPU is shared with an LLM or speech-to-text, leave that much free for Ground Control.

If the GPU has memory to spare, set `KEEP_GPU_MEMORY` to `true`. Isolation then keeps about 0.75 GB more memory between commands. Each command is done about 0.03 s sooner on a GB10 and 0.01 s sooner on an RTX 4000. Vision Match's models keep about 0.25 GB more between looks. A look with every feature then takes 17 ms instead of 31 ms on a GB10 and 17 ms instead of 21 ms on an RTX 4000.

### Intel GPU

Add `--device /dev/dri` and use the `openvino` tag:

```bash
docker run -d --name ground-control --network host --device /dev/dri \
  -e HA_URL=http://homeassistant.local:8123 \
  -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control:openvino
```

The OpenVINO image runs Vision Match's models on an Intel GPU with OpenVINO. A look takes a fraction of its time on the CPU (see [Performance Benchmarks](../README.md#performance-benchmarks)). Voice Match runs on the CPU as in the CPU image: the speaker models, the Wake Word Aligner and Isolation are as fast or faster there.

The OpenVINO image needs:

- An Intel GPU from the 11th generation of Core processors (Xe graphics) or newer. It can be integrated, including the one in an Intel N100, or an Arc card. It was tested on an Iris Xe and on an Intel N100.
- The GPU passed to the container with `--device /dev/dri`. The host needs nothing installed for it beyond its normal graphics driver.
- An amd64 machine.

The vision models take about 0.3 GB of memory on the GPU, or 0.5 GB with every model on Accurate. An integrated GPU takes that from the machine's own memory.

Each model is compiled for the GPU the first time it runs, which takes a few seconds. The compiled models are kept in the `openvino` folder of the data volume, so later starts load them in under a second. The folder takes about 0.1 GB and can be deleted at any time.

The OpenVINO image runs Vision Match on the GPU or not at all. If it can't use the GPU, it stops and logs a line that starts with `OpenVINO can't start`. The line gives the reason. For example, no GPU is visible to the container or the GPU isn't an Intel one. With more than one GPU in the machine, pass only the Intel GPU's render device, such as `--device /dev/dri/renderD128`, or set `OPENVINO_DEVICE` to the one to use, such as `GPU.1`.

### AMD GPU

Add `--device /dev/dri` and use the `vulkan` tag:

```bash
docker run -d --name ground-control --network host --device /dev/dri \
  -e HA_URL=http://homeassistant.local:8123 \
  -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control:vulkan
```

The Vulkan image runs the models on a GPU through Vulkan, the graphics interface that every current GPU driver has. It is meant for AMD GPUs, which the CUDA image can't use. The speaker models, Isolation, Face Detection, Hand Gestures and Face Recognition can all run on the GPU. The Wake Word Aligner runs on the CPU, as in every image.

The Vulkan image needs:

- A GPU with a Vulkan driver in Mesa. For an AMD GPU that is the RADV driver, which is in the image.
- The GPU passed to the container with `--device /dev/dri`. The host needs nothing installed for it beyond its normal graphics driver.
- An amd64 machine running Linux. Docker on Windows doesn't give a container the GPU's own driver.
- About 2 GB more memory than the CPU image with every model on the GPU, and about 1 GB on the GPU.

#### Each Model Runs Where It Is Faster

When a model loads, Ground Control runs it on the GPU and on the CPU with the same input. It compares the results and the time:

- If the results differ, the model runs on the CPU and the log gives the reason. This protects against a driver that computes a model wrong. For Isolation, the model turns off.
- If the GPU is not faster than the machine's CPU, the model runs on the CPU. **Settings** shows where each one runs.

A small integrated GPU is slower than its machine's CPU for every model, so everything runs on the CPU there. This is the case for the Radeon graphics of a Ryzen 9000 desktop processor and for an Intel Iris Xe. For an Intel GPU, the [OpenVINO image](#intel-gpu) is faster for Vision Match.

Face Detection and Hand Gestures are Accurate by default when the GPU runs the Accurate model in 50 ms or less. Ground Control times both at start and the log shows the result. On a slower GPU the defaults are the CPU image's, which is Light for both. A model you choose under **Settings > Vision Match** stays as chosen.

#### How Fast It Is

These are the times on an RTX 4000 through Vulkan, next to its i9-13900H on four threads. An NVIDIA GPU is faster still with the CUDA image, so use that one for NVIDIA.

| Model | GPU through Vulkan | CPU |
| --- | --- | --- |
| Accurate speaker model, a 1 second check | 7 ms | 15 ms |
| Isolation, a 5 second command | 0.12 s | 0.57 s |
| Face Detection, Light and Accurate | 8 and 13 ms | 17 and 45 ms |
| Hand Gestures, Light and Accurate | 5 and 26 ms | 22 and 290 ms |
| Face Recognition, Light and Accurate | 3 and 14 ms | 6 and 30 ms |

The image has not been timed on a Radeon card yet.

#### If the GPU Stops Answering

Everything that touches the GPU runs in helper processes, one for each model, and Isolation is sent to the GPU in short jobs. A GPU driver cancels a job that runs for too long, which is 2 seconds with Linux's AMD driver, so no job is allowed to get near that.

If the GPU is lost anyway, such as after a driver reset, Ground Control goes on:

- A voice check or a Vision Match look that was under way is done on the CPU.
- A command that Isolation was working on goes to the assistant as Trimming left it, and its session shows **Isolation not used**.
- The helper process starts again and the model is back on the GPU a few seconds later.

#### More Than One GPU

If Ground Control finds no GPU, it stops and logs the reason. With more than one GPU in the machine, pass only the render device of the one to use, such as `--device /dev/dri/renderD128`, or set `VULKAN_DEVICE` to part of its name, such as `Radeon RX`. The start of the log names the GPU it found.

#### What It Was Tested On

- An NVIDIA RTX 4000 through NVIDIA's Vulkan driver: every model on the GPU. The 2,357 frames of the kiosk test set read the same as on the CPU, and 120 recorded voice sessions gave the same events.
- Mesa's RADV driver with the integrated Radeon of a Ryzen 9 9950X3D: every model gives the CPU's results. That GPU is slower than its CPU, so the models run on the CPU there.
- Mesa's Intel driver with an Iris Xe: the same.

It has not been tested on a Radeon card yet.

### Mac with Apple Silicon

Ground Control runs with the CPU image in [Docker Desktop](https://www.docker.com/products/docker-desktop/) or [OrbStack](https://orbstack.dev) on a Mac with an Apple M-series chip. Docker on a Mac can't use the Mac's GPU, so everything runs on the CPU. An M-series chip is fast enough for every feature. On a MacBook Pro M5, all Voice Match features together add about 0.8 s to a 5 s command. A look with every Vision Match feature takes 70 ms.

1. Install Docker Desktop or OrbStack.
2. Start Ground Control with its port published:

    ```bash
    docker run -d --name ground-control --restart unless-stopped -p 10900:10900 \
      -e HA_URL=http://homeassistant.local:8123 \
      -v ground-control:/data ghcr.io/jxlarrea/kiosk-satellite-ground-control
    ```

3. Open `http://localhost:10900` and choose **Log in with Home Assistant**.
4. In Kiosk Satellite's Ground Control settings on each kiosk, enter the Mac's address (for example `192.168.1.20:10900`) under **Address**.

Ground Control on a Mac isn't discovered automatically. Docker runs it in a small Linux VM, so its mDNS announcement doesn't reach your network. Docker Desktop's host network option doesn't change that. Give the Mac a fixed address in your router so the address in each kiosk keeps working.

Ground Control picks the Accurate speaker model and the Accurate Face Recognition model automatically on an Apple chip. Keep the Mac from sleeping. On a Mac mini, the setting is under **System Settings > Energy**. On a MacBook, it is under **Battery > Options**. While the Mac sleeps, the kiosks use their own microphone and Voice Match is unavailable. The Vision Match features on Ground Control wait until it answers again.

### Skip the Login

Set `HA_TOKEN` to a long-lived access token to open the web app without logging in with Home Assistant. Create one in Home Assistant under your profile, on the **Security** tab, under **Long-lived access tokens**. The token stays in Ground Control and is never sent to the browser. If Home Assistant refuses the token, the Sign in page says so and shows **Log in with Home Assistant** instead.

```bash
  -e HA_TOKEN=YOUR_LONG_LIVED_TOKEN \
```

> **Warning:** With `HA_TOKEN` set, anyone who can reach port 10900 can open Ground Control and change its people, kiosks and settings without logging in. Only set it on a network you trust. Kiosks still sign in with their own token.

### Environment Variables

| Variable | Default | What it does |
| --- | --- | --- |
| `HA_URL` | `http://homeassistant:8123` | Your Home Assistant address. Kiosk tokens are checked there and you log in to the web app there. |
| `HA_TOKEN` | none | A long-lived Home Assistant access token. The web app opens without logging in for anyone who can reach the port. See [Skip the Login](#skip-the-login). |
| `MODEL` | `auto` | The speaker model: `accurate`, `light` or `auto`. `auto` picks Accurate on an NVIDIA GPU, an Apple M-series chip or an x86 CPU with more than 4 cores and Light otherwise. The Vision Match models are chosen in the app under **Settings > Vision Match**. |
| `THREADS` | half the CPUs, 1 to 4 | Threads per check. |
| `KEEP_DAYS` | `7` | Days saved sessions are kept. |
| `KEEP_AUDIO` | `true` | Keep each session's audio. |
| `PORT` | `10900` | The port for kiosks and the web app. |
| `MDNS` | `true` | Advertise the service so it is discovered automatically on the network. |
| `MDNS_ADDRESS` | the main network's | The address kiosks get. See [Kiosks on Another Network](kiosks.md#kiosks-on-another-network). |
| `DATA_DIR` | `/data` | Where people, voiceprints, faces, kiosks and sessions are stored. |
| `UPDATE_CHECK` | `true` | Look for a new Ground Control version on GitHub every 12 hours and show it on Overview. |
| `OPENVINO_DEVICE` | `GPU` | OpenVINO image only. The GPU to use when the machine has more than one, such as `GPU.1`. See [Intel GPU](#intel-gpu). |
| `VULKAN_DEVICE` | the first GPU | Vulkan image only. Part of the name of the GPU to use when the machine has more than one, such as `Radeon RX`. See [AMD GPU](#amd-gpu). |
| `KEEP_GPU_MEMORY` | `false` | CUDA image only. Isolation and the vision models keep their GPU memory between runs instead of handing it back. See [GPU Memory](#gpu-memory). |

## Updating

**Home Assistant app:** update it from the app's page when Home Assistant offers a new version.

**Docker:** run `docker compose pull && docker compose up -d` or pull the image again and recreate the container. The data volume is kept.

Ground Control checks GitHub for a new version a minute after it starts and every 12 hours after that. When there is one, a card at the top of **Overview** shows the version and links to what's new and to the repository. The check only asks GitHub for the list of versions. Set `UPDATE_CHECK=false` to turn it off. The Home Assistant app skips this check because Home Assistant offers its updates.

To move Ground Control to another machine or between the app and Docker, see [Backup and Moving](backup.md).

Each Ground Control version works with the Kiosk Satellite version released alongside it. Update the kiosks too when you update Ground Control.

## Next Steps

1. [Connect and approve your kiosks](kiosks.md).
2. [Add people and enroll their voices and faces](people.md).
3. [Turn on Voice Match on each kiosk](voice-match.md#on-the-kiosks).
4. [Turn on Vision Match on each kiosk with a camera](vision-match.md#on-the-kiosks).
