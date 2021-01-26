#!/bin/bash

set -ex

echo "------Train efficientnet b5------"
python main_b5.py --root /home/yeonsoo/dfdc_deepfake_challenge-master/data/original_data/

echo "------Train efficientnet b6------"
python main_b6.py --root ./data

echo "------Train efficientnet b7------"
python main_b7.py --root ./data
