#!/usr/bin/env bash
# One-time setup of the montage toolkit in a fresh container.
set -e
cd "$(dirname "$0")"
pip install -q faster-whisper "opencv-python-headless<5" pillow numpy scipy huggingface_hub
# fonts (Inter, SIL OFL) if the system doesn't have them
if ! fc-list | grep -q "Inter ExtraBold"; then
  mkdir -p ~/.fonts && cp assets/fonts/*.otf ~/.fonts/ && fc-cache -f >/dev/null
fi
# speech model (~1.6 GB) into ./models
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from faster_whisper import WhisperModel
WhisperModel('large-v3-turbo', device='cpu', compute_type='int8', download_root='models')
print('whisper model ok')
from huggingface_hub import hf_hub_download
hf_hub_download('opencv/face_detection_yunet', 'face_detection_yunet_2023mar.onnx', local_dir='models/yunet')
print('yunet ok')
PY
