# RNSA surrogate baseline for TopAneu26

RSNA 2025 1위 해법의 핵심이었던 **vessel anatomy auxiliary learning**, **vessel-masked
pooling**, **aneurysm segmentation을 강하게 둔 multi-task loss**를 TopAneu26에 맞게 압축한 독립
baseline입니다. 이 저장소는 cache 생성부터 학습·resume·단일-volume 추론까지 필요한 파일만
포함합니다.

이 폴더의 목적은 무거운 3-model cascade를 바로 재현하는 것이 아니라, vessel-aware formulation이
단일 53-class nnU-Net baseline보다 실제 validation metric을 개선하는지 빠르게 검증하는 것입니다.

## 모델

```text
angiography + modality + z/y/x coordinates (5 ch)
                         │
                    3D U-Net
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  binary aneurysm   53-way location  37-way vessel aux
     segmentation     segmentation     (half resolution)
          │                             │
          └──── vessel-masked bottleneck pooling ────┐
                                                     ▼
                                      52-location multi-label
                                      + aneurysm presence
```

- `aneurysm_logits`: class와 무관하게 작은 병변 recall을 확보하는 주 segmentation head
- `location_logits`: Task 2의 0–52 위치 mask를 생성하는 head
- `vessel_logits`: 공개 silver vessel mask를 학습할 때만 쓰는 auxiliary head
- `location_presence_logits`: Task 1을 위한 52개 multi-label head
- `aneurysm_presence_logits`: scan/patch-level 유무 head

Loss 기본 가중치는 RNSA 해법처럼 binary aneurysm auxiliary task를 가장 강하게 두고, vessel과
classification을 보조 신호로 사용합니다. 좌우 flip은 image만 뒤집지 않고 52-location label,
36-vessel label, global x-coordinate를 함께 교환합니다.

## 설치와 검사

CUDA에 맞는 PyTorch를 먼저 설치한 clean environment에서 실행합니다.

```bash
cd TopAneu26_baselines
python -m pip install -e '.[train]'
```

## 1. Physical-space cache 생성

본 학습은 원본 NIfTI를 직접 읽지 않습니다. 먼저 모든 영상을 0.6 mm isotropic으로 resampling하고
atomic `.npy` cache를 생성합니다. Image는 linear interpolation, location/vessel label은
nearest-neighbor를 사용하며, downsampling에서 사라진 tiny lesion component는 원래 physical
center에 1-voxel seed를 복원합니다. 이 구조와 로깅 계약은
[TopAneu26 Prototype](https://github.com/choe-git/TopAneu26_Prototype)의 pipeline을 참고했습니다.

```bash
RUN_DIR=/path/to/runs/RNSA_surrogate/$(date +%Y%m%d_%H%M%S)

python scripts/prepare_cache.py \
  --source /path/to/topaneu_release \
  --split-csv split.csv \
  --run-dir "$RUN_DIR" \
  --spacing 0.6 0.6 0.6

python scripts/inspect_cache.py \
  --cache "$RUN_DIR/cache" \
  --deep \
  --output "$RUN_DIR/cache_report.json"
```

`cache/index.json`이 cache의 commit marker입니다. 중간에 작업이 끊겨 index가 없으면 학습이
시작되지 않습니다. `--overwrite`는 같은 cache를 의도적으로 다시 만들 때만 사용합니다.

## 2. 학습

cache를 만든 동일한 `RUN_DIR`에서 학습합니다.

```bash
python scripts/train.py \
  --config configs/baseline.yaml \
  --run-dir "$RUN_DIR" \
  --device cuda
```

학습은 cache spacing과 config spacing이 다르면 실패합니다. Cache manifest SHA-256과 학습 계약은
checkpoint에 저장되며, 다른 cache/config로 잘못 resume하는 것도 차단합니다.

### Run과 logging 구조

```text
RUN_DIR/
├── cache/
│   ├── index.json
│   └── cases/<case_id>/{image,location,vessel,instances}.npy
├── cache_report.json
├── model/
│   ├── config.json
│   ├── environment.json
│   ├── inputs.json
│   ├── model.json
│   ├── status.json
│   ├── training_log.txt
│   ├── checkpoint_latest.pth
│   ├── checkpoint_best.pth
│   ├── checkpoint_epoch_XXXX.pth
│   └── metrics/{train,val,test}/epoch_XXXX.json
└── tensorboard/
```

TensorBoard에는 각 loss component, total loss, learning rate를 split별로 기록합니다. Validation과
best checkpoint는 EMA weight로 계산하고, latest checkpoint에는 model/EMA/optimizer/scheduler/
GradScaler/RNG state가 모두 들어갑니다.

```bash
tensorboard --logdir "$RUN_DIR/tensorboard"
```

중단된 run은 같은 위치에서 이어갑니다.

```bash
python scripts/train.py \
  --config configs/baseline.yaml \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --resume "$RUN_DIR/model/checkpoint_latest.pth"
```

Train patch는 lesion component 단위로 sampling하며 희귀 class와 작은 병변을 oversample합니다.
Negative patch의 일부는 vessel point를 중심으로 뽑아 혈관 주변 false positive를 학습합니다.

## 단일 volume 추론

```bash
python scripts/predict.py \
  --image /path/to/case_0000.nii.gz \
  --checkpoint "$RUN_DIR/model/checkpoint_best.pth" \
  --output /path/to/predictions \
  --device cuda
```

출력은 다음과 같습니다.

```text
predictions/
├── task1_locations/<case>.json
└── task2_masks/<case>.nii.gz
```

추론은 53채널 full-volume accumulator를 만들지 않고 overlap window별 최고 class confidence만
보존하므로 메모리 사용량이 volume 크기에 대해 작습니다. Threshold는 validation에서 조정해야 하며,
Grand Challenge의 `.mha` socket/container adapter는 연구 baseline 성능이 확인된 뒤 붙이는 범위로
남겨 두었습니다.

## 권장 비교 순서

1. 기존 nnU-Net baseline과 동일 split에서 validation 공식 metric 측정
2. `loss.vessel=0`과 `0.1` 비교로 anatomy branch의 순수 기여 확인
3. binary aneurysm head와 location head의 threshold sweep
4. 이득이 확인되면 coarse ROI locator와 model-driven hard-negative mining 순으로 확장

이 구현은 학습 가능한 baseline 코드이며 pretrained weight나 성능 수치는 포함하지 않습니다.
