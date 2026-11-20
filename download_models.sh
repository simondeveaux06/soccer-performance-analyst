#!/bin/bash
# Downloads the pretrained pitch-landmark model (Roboflow sports, YOLOv8-pose, 32 pitch keypoints).
# Source: https://github.com/roboflow/sports
set -e
cd "$(dirname "$0")"
mkdir -p models
pip install -q gdown
gdown -O models/football-pitch-detection.pt "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"
echo "Done: models/football-pitch-detection.pt"
