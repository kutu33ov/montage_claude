# montage_claude — how to cut shorts for the owner

The owner records Russian-language talking-head videos (topic: AI, social networks, content),
sends a Google Drive link, and expects finished vertical shorts + titles + covers.
Talk to the owner in Russian, casually, short messages.

## ALWAYS ask before cutting (every new video)
After `prep.py` (so you can already propose concrete moments), and BEFORE rendering anything, ask the owner
these six questions — use AskUserQuestion (max 4 per call → two calls), give concrete options with your
recommendation first, based on the transcript:
1. **Длина** — how many seconds per short (e.g. 30 / 45 / 60 / as the thought needs) and how many shorts.
2. **Ретушь** — where: `nose_tip`, `nose`, `forehead`, `cheek_left`, `cheek_right`, `between_brows` (scripts/skin.py
   ZONES) + general skin smoothing strength (0 = off, 0.5 = light default). Offer a 9:16 before/after screenshot.
3. **Графики** — insert charts? Propose the exact numbers/claims from the speech for each card (never invent data,
   show the source the owner named).
4. **Картинки** — insert images? Which ones (owner sends them or names them; nothing downloaded from random sites).
5. **Бесконечное видео** — make it loop endlessly? If yes, pick an ending that flows into the opening
   (e.g. the last phrase leads into the first sentence) and set `"loop": true`.
6. **Переходы** — transition effects or plain cuts? (He dislikes glitch and shake; offer whip / zoom / flash / none.)
Then render with the answers. If he already answered some of these in his message, don't ask them again.

## THE STANDARD (owner, Nov 20 2026: «95% всё хорошо, без ошибок, склейка очень хорошая — делай так всегда»)
The 12-video outdoor batch (projects v7_*) is the reference. What made it good — always do this:
1. Transcribe with VAD + no conditioning (scripts/transcribe.py) so every retake is a separate line.
2. Read the whole transcript; for each video keep ONE complete thought. Short clips (15–35 s) that are
   complete are shorts too — never skip a video because it is short; ask if unsure.
3. Keep only the LAST complete take of every phrase; drop abandoned starts and «продолжение следует».
4. Prefer few large parts in the original order; cut only whole phrases. Let make_short cut the pauses.
5. Before rendering, print and READ the final caption text of every short end to end: no repeats, no
   cut words, no missing words, the meaning flows from hook to conclusion.
6. Render with: fps 60, crf 17, punch-in 1.12 + drift 0.06, no transitions, light-red terms, quiet music,
   retouch {skin 0, tone 0.2, nose_red, spots}. Native vertical sources are used as is (no upscaling).
7. **Every video starts smoothly, without abrupt cuts:** the first ~3 s have no jump cuts and no punch-in
   (`smooth_start`, default 3.0 s — small pauses in the opening stay in).
8. CHECK THE SOURCE FIRST, before any work: native phone files (.MOV/HEVC, vertical, real 60 fps) give the
   good result. The Nov-20 indoor batch (v6_*) came as re-encoded 1920×1080 H.264 «60 fps» with every frame
   doubled and an IRREGULAR duplicate pattern — the judder was baked into the source; no render setting fixed
   it (the owner confirmed: «проблема была в исходниках»). Count duplicate frames over ~120 source frames;
   if there are many / the pattern is irregular, TELL THE OWNER and ask for the original phone files
   (upload the .MOV to Drive as is, no editor/converter) instead of cutting it.
9. Run ONE render queue at a time (two runners once wrote the same file and corrupted it); after
   rendering, decode-check every file before sending (`ffmpeg -v error -i f -f null -`).

## Setup in a fresh session
1. `bash setup.sh` (pip deps, Inter fonts, ~1.6 GB whisper model into `./models`). In the `montage_claude`
   cloud environment this runs automatically at session start.
2. Network hosts needed (already allowed in the `montage_claude` and Default environments):
   `huggingface.co`, `cdn-lfs.hf.co`, `us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`, `cas-server.xethub.hf.co`,
   `drive.google.com`, `drive.usercontent.google.com`.
3. The Google Drive connector can't list/write files (missing permission) → download by link
   (`drive.usercontent.google.com/download?id=…&export=download&confirm=t`); the owner must share
   "anyone with the link". A Drive *folder* link: fetch the folder page with curl and parse file ids from `_DRIVE_ivd`.

## Pipeline
```
python3 scripts/prep.py <drive link> projects/<name>      # download, transcript, pauses, faces, config.json
# read projects/<name>/transcript.txt, choose shorts, fill "shorts" in config.json
python3 scripts/make_short.py projects/<name>/config.json <short_id>   # renders out/<short_id>.mp4
python3 scripts/pick_frames.py src.mp4 config.json <short_id> sheet.jpg 8   # best cover frames
python3 scripts/cover.py covers.json                        # covers
```
A short is `{"id", "parts": [[a,b], ...], "start", "end", "hook"}` (start/end = first/last part bounds;
`hook` = title shown in the first 3 s, `\\N` = line break). `parts` lets you stitch pieces and drop repeats.
Per-short `"fixes"` map recognised words to the right spelling. Source is assumed 1920×1080 landscape.
Render one short at a time (≈1–1.5 min each); the frame pipeline keeps memory low.

## Effects (scripts/fx.py) — full option list in `examples/short_example.json`
Per short in config.json (anchor by `"word"`, source time `"at"` or output time `"out"`):
- `emphasis` — big pop-up text with pop sound: numbers, key claims («53%»). `"shake": true` exists but the
  owner finds shake often inappropriate — don't use it unless asked.
- `inserts` — image card (white rounded frame, shadow, slight tilt) with pop-in + pop sound. The owner LIKES
  charts: when the speech has numbers/stats, add a generated infographic with `"card": {...}` instead of `"img"`
  (kinds `bars`, `stat`, `versus`, `columns` — see `scripts/card.py`). `pos: "auto"`
  fits it above the head (may cover hair, never the eyes) or between chin and captions.
  Images go to `assets/images` (or any path). Never invent facts on a card; label sources honestly.
- `transitions` — `flash` | `whip` | `zoom` | `glitch` | `shake`, each with a matching sound.
  The owner does NOT like `glitch` and finds `shake` usually out of place — prefer `whip`, `zoom`, `flash`.
  Auto: a `whip`+whoosh only where the short jumps > 30 s of source (`fx.part_gap`); trimmed repeats stay plain cuts.
- `clips` — someone else's video inside the short: `{"file": path, "from": 20, "to": 35, "word"/"at"/"out": anchor,
  "layout": "full" | "split" | "card", "audio": "mute" | "mix", "audio_db": -8, "label": "Simons & Chabris, 1999"}`.
  `full` = clip over a blurred copy of itself, `split` = clip on top / speaker below (best when he comments on it),
  `card` = small framed clip above the head. Always label the source; keep foreign footage short (Content ID).
- `sfx` — extra sounds: whoosh, swipe, pop, click, ding, riser, boom, glitch (`assets/sfx`).
- `music` — `{"file": "auto", "volume_db": -22, "duck": true}`; "auto" takes the owner's own track from
  `assets/music` first, the synthetic `placeholder_lofi_88bpm.wav` only if there is none. Ducked under the voice.
  The owner will write his own music — always prefer his tracks.
- `lut` — `warm` | `cool` | `contrast` | `film` (`assets/luts`, regenerate with `scripts/gen_luts.py`).
- `retouch` — `{"skin": 0.5, "nose_red": true, "warmth": 2.5, "spots": ["nose_wide"], "zones": []}`
  (global or per short; covers take it too). Video uses `skin.Retoucher`: YuNet face + landmarks EVERY frame
  (works on 3/4 and side views — the frontal Haar detector missed them and the pimple showed), smoothed and
  held through short dropouts; the nose area is centred on the nose-tip landmark and pulled to the CHEEK tone
  (+warmth so it's never pale); small red dots removed (`spots`). Extra redness zones (`forehead`,
  `cheek_left`…) via `zones`. The owner checks the nose tip frame by frame — verify with a nose contact
  sheet over the whole short (uniform frames + fastest-motion frames) before sending.
- `loop` — `true`: endless video. Last segment uses the first segment's zoom, HARD cut back to the start (the
  owner disliked the dissolve; `"loop_xfade": 0.35` re-enables it), music has no audible fade, hook title hidden (`"hook_in_loop": true` to keep it). The CONTENT
  must loop too: end on a phrase that leads into the opening line (e.g. end «…ролик, который тоже начался
  с вопроса, рассчитанного на то, чтобы тебя зацепить» → start «Ты заметил, что интернет…»).
- `prefilter` — ffmpeg filter chain applied to the source first. Low-light / evening footage (owner liked it):
  `hqdn3d=5:4:8:6,curves=all='0/0 0.08/0.12 0.25/0.36 0.6/0.72 1/1',eq=saturation=1.06,unsharp=5:5:0.45`
  (denoise + lift shadows/mids, blacks stay black — plain gamma made a grey veil). In low light the nose retouch
  does NOT work (owner) — set `"skin": false` (no retouch at all) for evening/night footage.
- `fx.progress_bar`, `fx.watermark` ("@nick") — off by default.
- `terms` — list of terms (global or per short) coloured light red (`term_color`, default FF6B6B) in the captions;
  word forms match by stem, multi-word terms colour every word. The owner wants terms highlighted.
- `drift` — slow push-in within every segment (e.g. 0.06) = «больше динамики» without transitions.
  Owner's current default (Nov 2026): punch-in zoom on cuts + drift 0.06, NO transitions (`fx.auto: false`).
- `fps` / `crf` / `maxrate` — 60 for 60 fps phone footage; output is BT.709-tagged, capped at 25 Mbps.
End every part at least ~0.25 s after the last word's end time (whisper `e`), never on it — otherwise
the last word gets clipped («подумай» was cut off).
Taste: 1–3 accents per short (one emphasis, one insert/glitch, a ding on the punchline). Don't put a sound
on every cut. All bundled sounds/music/LUTs are synthesised by `gen_sfx.py` / `gen_luts.py` — royalty-free.

## What the owner wants (learned the hard way)
**Choosing shorts**
- Each short = one complete thought: strong hook at the start, a *finished* ending. Never end on a teaser
  ("и вот это проблема номер один", "продолжение следует").
- No repetitions inside a short (same phrase/term said twice → cut one). Use `parts` to stitch.
- Watch for retakes: whisper may skip a repeated take (a word stretched over many seconds = suspicious) —
  re-transcribe that stretch alone and pick the better take. transcribe.py now uses VAD and no previous-text
  conditioning so retakes show up as separate lines (Nov 2026: «Но почему мы снова и снова…» was said twice
  and slipped into a short). Before rendering, READ the final caption text of every short end to end:
  no repeated phrases, no sentence cut mid-way, every part ends on a full phrase, the meaning flows.
- The owner often starts the sentence over when he stumbles: always keep only the LAST complete take,
  drop the abandoned starts.
- Never leave pieces with no speech: make_short drops keep-segments without words (a 1 s silent piece with
  a punch-in looked like a glitch to the owner).
- ≤60 s preferred; up to ~70 s is fine if the thought needs it.
**Edit**
- Pauses cut by audio silence (not whisper timestamps). Punch-in zoom 1.12 toggles on jump cuts.
- Subtitles: white, thin black outline, 1–3 words, lowercase, Inter ExtraBold, lower third.
  Prepositions/short conjunctions (в, к, на, с, по, и, а, не…) are NEVER the last word of a caption — they move
  to the next caption with their word; particles (же, ли, бы) stay with the previous word. Lines longer than
  18 chars shrink the font instead of wrapping. (Implemented in make_short.py — keep it that way.)
- Retouch: light skin smoothing (strength 0.5) + redness fix only in the zones he names (default: nose tip),
  matched to the surrounding skin tone (it must never look pale). The owner dislikes heavy/plastic smoothing.
- Loudness −14 LUFS.
**Covers (1080×1920)**
- Plain natural frame (with the same retouch) + title. NO colour grading, NO purple look, NO cut-out/outline,
  NO emoji. Soft dark fade at the bottom only.
- Title: sentence case, Inter SemiBold ~130 px, white, 2–3 short lines bottom-left (≤13 chars per line so all
  covers have the same big size), the key word on a violet brush stroke.
- Title text must be punchy and understandable WITHOUT watching the video, addressed to the viewer
  (good: «Ты споришь с ботом», «Половина интернета — уже боты», «Все блогеры копируют друг друга»;
  bad: «Неделя работы или 10 копий?» — unclear).
- Frame: sharpest frame, face frontal, eyes INTO the camera, pleasant face (mouth closed or slight smile, no
  mid-word grimace, no teeth). Use `pick_frames.py`, look at the sheet, and if unsure offer 3 numbered options.
**Delivery**
- Send each short with SendUserFile as soon as it's rendered (don't batch); check a few frames first.
- Show visual changes as 9:16 screenshots / before-after.
- For every short give a title (+ one alternative). Offer descriptions/hashtags.
- Remind that the container is temporary — the owner downloads the files.

## Assets
`assets/sfx`, `assets/music`, `assets/luts`, `assets/overlays`, `assets/images` — owner's effect packs
(whooshes, pops, music beds, .cube LUTs for `ffmpeg lut3d`, overlays with alpha, memes/logos). Use them when present.
Heavy packs may live in a shared Drive folder instead — ask for the link.
