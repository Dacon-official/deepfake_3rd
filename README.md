# aihub_dfdc_challenge

This repository contains the starter code for the "[딥페이크 변조 영상 탐지 AI 경진대회](https://dacon.io/competitions/official/235655/overview/)".
![Main  SNU_DFDC](https://user-images.githubusercontent.com/13975787/96542402-fa841f80-12dc-11eb-95a6-baceb25ae80c.png)

## Background
* 최근 딥페이크 변조 영상이 대중의 화제를 일으키고, 각종 AI 기반 얼굴 합성/변조 기술이 다양한 분야에서 관심을 받으면서 해당 영상 탐지에 대한 필요성이 대두되었습니다. 변조/합성 분야가 상대적으로 높은 관심을 받으면서 활발히 연구되는 반면, 변조/합성 탐지 연구는 상대적으로 적은 관심을 받고 있습니다. 

* 본 대회에서는 “AI hub 데이터 구축사업”의 일환으로써 “딥페이크 변조 영상 탐지”를 목적으로 합니다. 기존 변조/합성 탐지 연구 레퍼런스 데이터가 서양인 위주로 구성되고, 적은 데이터 양을 가지며, 탐지 방해 요소를 고려하지 않고 제작된 점 등의 문제점 해결하는 데이터를 제안함으로써 “AI hub 데이터 구축사업“ 홍보 및 데이터 사용 활성화를 목적으로 합니다.

## Prerequisites
- Ubuntu 18.04 LTS
- Python3
- CPU or NVIDIA GPU + CUDA CuDNN

## Training
To train a model, use the `train.sh` script.

## Inference
To infer a model, use the `infer.sh` script.

Make sure your `path_to_dataset` is appropriate.  

### Caution

The output of the model depends on the Pytorch version you use.

Our result was obtained by using different Pytorch version as follows:
- `Pytorch 1.4.0` for the 'efficientnet-b5' model
- `Pytorch 1.5.0` for the 'efficientnet-b6' model
- `Pytorch 1.4.0` for the 'efficientnet-b7' model

## Contributors
- Hyeongjun Kwon (kwonjunn01@naver.com)
- Jinhyun Jang (wlsugs0615@naver.com)
- Yeonsoo Park (fhfkdshfkd@naver.com)
