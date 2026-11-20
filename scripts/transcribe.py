import sys, json, os
MODELS = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from faster_whisper import WhisperModel
src, out = sys.argv[1], sys.argv[2]
m = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8', download_root=MODELS, cpu_threads=4)
import wave, numpy as np
wf=wave.open(src); audio=np.frombuffer(wf.readframes(wf.getnframes()),dtype=np.int16).astype(np.float32)/32768
segs, info = m.transcribe(audio, language='ru', word_timestamps=True, vad_filter=True, beam_size=5,
                         condition_on_previous_text=False)   # VAD + no conditioning: retakes are no longer swallowed
res = []
for s in segs:
    res.append({'start': s.start, 'end': s.end, 'text': s.text.strip(),
                'words': [{'w': w.word.strip(), 's': w.start, 'e': w.end, 'p': w.probability} for w in s.words]})
    print(f"[{s.start:7.2f}-{s.end:7.2f}] {s.text.strip()}", flush=True)
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
