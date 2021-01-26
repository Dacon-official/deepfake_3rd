#!/bin/bash

set -ex

echo "------Inference------"
python3 inference.py --data_root ./leaderboard
# You have to choose the appropriate data root for the inference

