# People

Navigate to **People** in the Ground Control app.

This page lists everyone Voice Match and Face Recognition can recognize. It holds each person's voice, face, color and the things "my" means for them. The list on the left shows each person, whether their voice and face are enrolled and how many device models the voice is enrolled on.

## Adding a Person

1. Choose **Add a person** and enter their name. The kiosks and the assistant use this name for them.
2. Enroll their voice on each device model they talk to. See [Enrollment](#enrollment).
3. For Face Recognition, enroll their face on a kiosk with a camera. See [Enrolling a Face](#enrolling-a-face).

A person without a voice or a face is never recognized. Someone with only a face is recognized by Face Recognition but never by their voice. They can still be picked on a kiosk under **Answer > Chosen people**.

## Enrollment

A voice is recognized best on the device models it was enrolled on, since the same voice sounds different through an Echo Show and an Echo Dot. Enrollment always records with a kiosk's own microphone, so enroll each person once on every device model they talk to.

1. On the person's page, **Device models** in the **Voice Match** card lists every device model in the house and the seconds of speech enrolled on each. Choose **Enroll** on a model or pick a kiosk under **Enroll on a kiosk** and choose **Enroll voice**.
2. The kiosk shows the text to read full screen in the kiosk's language. The Ground Control app shows the same text, so it can also be read from a phone.
3. Read it aloud for about 45 seconds:
    - Talk the way you give commands instead of using a reading voice.
    - Start close to the kiosk and then move across the room.
    - Face away from the kiosk now and then.

The sentence to read fills as it is spoken and moves on when it is full or after a pause. When another voice such as a TV talks, that part is left out and a banner shows it. The enrollment ends once 45 seconds of speech are counted. Choose **Enroll next** to go to the next device model the person isn't enrolled on.

If you stop an enrollment early with at least 10 seconds of speech, what was recorded is kept. The wake word is off during an enrollment, so reading the text never starts a voice command.

You can only enroll on kiosks with **Allow enrollment here** turned on in Kiosk Satellite under **Settings > Ground Control > Voice Match**. A kiosk busy with a voice command or an alarm can't start an enrollment until it is done.

### Recordings

**Recordings** at the bottom of the **Voice Match** card opens the list of enrollments with the kiosk each was recorded on and how much speech it has. Deleting one rebuilds the person's voice without it.

### Adding a Voice from a Session

Open a session where the person talked on the **Sessions** page and use **Use this voice** to add their speech from it to their voice. Only the user's speech is added. The other voices that were left out are not. This is useful for a device model nobody enrolled on yet or for a voice that has changed. See [Sessions](sessions.md#use-this-voice).

A voice is only ever made from audio you chose to add: an enrollment or a session's speech added with **Use this voice**. Nothing is added to it in the background. If a voice has changed or a kiosk in a new room doesn't recognize someone well, enroll once more on that kiosk.

## Face

Face Recognition matches the face of whoever is looking at a kiosk against the faces enrolled here. Turn it on for each kiosk with Vision Match in Kiosk Satellite under **Settings > Ground Control > Vision Match**. The kiosk's Face sensor in Home Assistant then shows who is recognized.

### Enrolling a Face

A face is enrolled with a guided capture of about half a minute through a kiosk's own camera. Start it on the kiosk or from the person's **Vision Match** card here. To start it here, pick the kiosk under **Enroll on a kiosk** and choose **Enroll face**. Only kiosks with Face Recognition on are offered. The kiosk then shows the steps:

1. Look at the screen.
2. Turn your head a little to one side.
3. Now a little to the other side.
4. Take a step back.

Each step needs a few good pictures of the face. When a picture isn't good, the kiosk shows what is wrong: no face, more than one face, come closer, turn more or less, too dark. A different face is left out, so someone walking past doesn't end up in the enrollment. The person's page shows the capture's progress as it happens, wherever it was started. It shows the kiosk, the step, the pictures taken and what to fix. To stop early, use **Stop** on the kiosk.

If the capture stops early after the first step, what was captured is kept. Enrolling on another kiosk or in other light helps the face be recognized there.

Ground Control keeps only small pictures of the face that are lined up and 112 pixels wide. Changing the Face Recognition model rebuilds the faces from them, so nobody has to enroll again. The **Vision Match** card shows them under **Face pictures**. **Captures** opens the list of captures with the kiosk each was taken on. Deleting a capture rebuilds the person's face without it.

### How a Face Is Recognized

A face looking at the kiosk is compared with every enrolled face. The face is named after two looks in a row above the **Threshold** under **Settings > Vision Match > Face Recognition** or after one very close match. The best match also has to be clearly ahead of the next person. Once named, the face is checked again every 2 seconds. When no face has looked at the kiosk for 3 seconds, nobody counts as in front of it.

Up to three faces looking at the kiosk are recognized at once. Each one is tracked from look to look by its position. The person at the kiosk is the one with the highest score. Someone else becomes the person at the kiosk only after scoring higher two checks in a row or when the person at the kiosk leaves. This keeps it from flipping between two people side by side. The kiosk's **Face** sensor shows the person at the kiosk and its **People in view** sensor shows everyone recognized. The kiosk's page and the vision sessions show the others in view too.

### Not a Lock

There is no check for a real person, so a photo or a video of an enrolled person held up to the camera can be recognized as them. Twins, siblings and young children can be mistaken for each other. Use the Kiosk Satellite settings that act on faces to skip steps for your family, not to keep anyone out, and don't unlock doors or disarm an alarm with the Face sensor or the face event in Home Assistant.

## Color

**Color on the kiosk** is the person's color: the name before a command on the voice overlay, the listening arc on a device with a light ring, the ring around their picture on the recognition cards and the person's marks in the Ground Control app.

## Home Assistant and Personal Commands

Each person can be linked to things in Home Assistant. As you type, the fields suggest matching entities and Music Assistant's playlists.

| Field | What it does |
| --- | --- |
| To-do list | "Add milk to my list" and "what's on my list" use it when this person says them. |
| Calendar | "What's on my calendar" reads it. |
| Playlist | "Play my playlist" plays it in Music Assistant. |
| Person | This person in Home Assistant. The People page and the kiosk's recognition cards show their Home Assistant picture. Speaker events and face events include it for automations. |

The kiosks receive these along with the voices, so "my" works on the kiosk with no extra step. Anyone without a list of their own uses the shared list. Kiosk Satellite's blueprints add the sentences to Home Assistant's own conversation agent and to LLM agents. See [Personal Commands](https://github.com/jxlarrea/kiosk-satellite/blob/main/docs/voice-match.md#personal-commands) in the Kiosk Satellite docs.

## Renaming and Forgetting

**Rename** changes the name everywhere including the kiosks.

**Forget** deletes the person, their voice, their face and all their enrollment audio and face pictures. Kiosks that answered only chosen people stop answering them. A kiosk whose chosen people are all gone answers nobody until people are chosen again. Its page on the Kiosks page shows this.
