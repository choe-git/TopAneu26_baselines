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
export USER_ROOT="/home/introai30/.apni/users/yhchoe"
export PROJECTS_ROOT="$USER_ROOT/projects"
export REPO_ROOT="$PROJECTS_ROOT/5_TopAneu/TopAneu_baseline"
export DATA_ROOT="$PROJECTS_ROOT/resources/topaneu_release"
export SPLIT_CSV="$REPO_ROOT/split.csv"
export RUN_ROOT="$PROJECTS_ROOT/runs/5_TopAneu/baseline"
export RUN_ID="$(date +%Y%m%d_%H%M%S)"
export RUN_DIR="$RUN_ROOT/$RUN_ID"

cd "$REPO_ROOT"
python -m pip install -e '.[train]'
```

Cache 생성부터 추론까지 같은 `RUN_ID`와 `RUN_DIR`을 유지합니다. 재접속 후에는 새 시간을 만들지
말고 기존 `RUN_ID`를 다시 지정해야 합니다.

## 1. Physical-space cache 생성

본 학습은 원본 NIfTI를 직접 읽지 않습니다. 먼저 모든 영상을 0.6 mm isotropic으로 resampling하고
atomic `.npy` cache를 생성합니다. Image는 linear interpolation, location/vessel label은
nearest-neighbor를 사용하며, downsampling에서 사라진 tiny lesion component는 원래 physical
center에 1-voxel seed를 복원합니다. 이 구조와 로깅 계약은
[TopAneu26 Prototype](https://github.com/choe-git/TopAneu26_Prototype)의 pipeline을 참고했습니다.

```bash
python scripts/prepare_cache.py \
  --source "$DATA_ROOT" \
  --split-csv "$SPLIT_CSV" \
  --run-dir "$RUN_DIR" \
  --spacing 0.6 0.6 0.6

python scripts/inspect_cache.py \
  --cache "$RUN_DIR/cache" \
  --deep \
  --output "$RUN_DIR/cache_report.json"
```

`cache/index.json`이 cache의 commit marker입니다. 중간에 작업이 끊겨 index가 없으면 학습이
시작되지 않습니다. `--overwrite`는 같은 cache를 의도적으로 다시 만들 때만 사용합니다.

Cache CLI 전체 계약:

| argument | 필수 | 의미 |
|---|---:|---|
| `--source PATH` | 예 | `images`, `location_masks`, `location_jsons`, `vessel_masks`를 포함한 release root |
| `--split-csv PATH` | 예 | train/val/test 분할 CSV |
| `--run-dir PATH` | 권장 | cache를 `RUN_DIR/cache`에 생성 |
| `--output PATH` | 대안 | cache 직접 경로; `--run-dir`와 동시 사용 불가 |
| `--spacing Z Y X` | 아니오 | 기본값 `0.6 0.6 0.6` |
| `--overwrite` | 아니오 | 기존 cache를 의도적으로 재생성 |

검사 CLI는 `--cache PATH`가 필수이고, `--deep`은 각 array header까지 검사하며,
`--output PATH`는 JSON report 저장 위치입니다.

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

## Winner-inspired vessel pretraining + 5-fold ensemble

Kaggle RSNA 1st-place write-up의 핵심을 TopAneu의 52개 location과 37-class vessel annotation에
맞춘 2-stage 경로입니다.

1. 기존 train/val split의 vessel mask로 residual 3D nnU-Net-style encoder/decoder와 vessel
   head를 단 한 번 pretraining합니다. train split으로 최적화하고 val split의 vessel loss로
   shared checkpoint를 선택합니다.
2. Vessel pretraining이 끝난 뒤, 원래 test split을 완전히 제외한 train+val case를 modality와
   52개 location 기준으로 multilabel-stratified 5-fold로 나눕니다.
3. 동일한 shared vessel EMA checkpoint의 encoder, decoder, half-resolution vessel head를
   5개 aneurysm fold 모델 모두에 이식합니다.
4. 52개 vessel-region token을 학습하고 location-aware Transformer로 location presence를
   예측하면서 aneurysm sphere/location segmentation을 함께 fine-tuning합니다.
5. test에서는 5개 fold의 voxel/location/presence 확률을 soft voting하고, 원본과 left-right
   flip 결과를 평균합니다. Flip 결과는 cache의 left/right label LUT로 원래 label에 복원합니다.
   Voxel location은 background를 제외한 52-class conditional softmax와 overlap consensus로
   결정합니다. Task 1은 single-patch maximum 대신 aneurysm evidence로 gate한 top-k patch
   평균을 사용해 한 noisy patch가 여러 location을 활성화하는 현상을 억제합니다.

현재 radical baseline은 shared encoder 뒤에 vessel decoder와 aneurysm decoder를 분리합니다.
Binary aneurysm loss는 Focal-Tversky + Dice + positive-weighted BCE를 사용하고, 52-class
location CE는 background를 제외한 실제 aneurysm voxel에서만 계산합니다. 추론에서는 binary
mask를 location confidence로 제거하지 않습니다. 먼저 3D connected component를 만든 뒤 component
내부 confidence-weighted vote로 위치 하나를 배정하며, scan당 confidence 상위 5개 component만
유지합니다. Task 1 label도 이 component에서 생성합니다.

`model.dual_decoder: true`는 checkpoint 구조를 변경하므로 이 설정으로는 새 RUN_DIR에서 다시
학습해야 합니다. 과거 checkpoint는 과거 config에 해당 항목이 없으므로 계속 읽을 수 있습니다.

여기서 nnU-Net-style은 외부 `nnUNetv2` CLI checkpoint를 불러오는 것이 아니라, 현재 cache와
aneurysm 모델이 encoder/decoder weight를 직접 공유할 수 있도록 repo 내부에 구현한 residual
3D U-Net pretraining stage를 의미합니다. Vessel pretraining 자체는 fold를 나누지 않으며,
multilabel-stratified fold는 이후 aneurysm fine-tuning에만 적용됩니다.

전체 과정은 한 명령으로 실행합니다.

```bash
python scripts/train_5fold.py \
  --run-dir "$RUN_DIR" \
  --config configs/baseline.yaml \
  --device cuda \
  --split test \
  --no-save-predictions
```

이 명령은 shared vessel pretraining, `folds.json` 생성, fold별 aneurysm fine-tuning, 5-fold
soft voting과 left-right TTA 공식 평가를 순차 실행합니다. 중단 후 같은 명령을 다시 실행하면
완료된 stage는 건너뛰고 미완료 stage는 `checkpoint_latest.pth`에서 재개합니다.

기본값은 shared vessel pretraining 1000 epoch와 fold별 aneurysm fine-tuning 150 epoch입니다.
전체 학습량은 vessel stage 1000 epoch와 aneurysm stage 750 epoch입니다.

Kaggle write-up처럼 5개 중 4개 fold만 ensemble하려면 다음처럼 지정할 수 있습니다.

```bash
python scripts/train_5fold.py \
  --run-dir "$RUN_DIR" \
  --config configs/baseline.yaml \
  --device cuda \
  --folds 0 1 2 3
```

주요 출력은 다음과 같습니다.

```text
RUN_DIR/
├── cache/
├── cache_report.json
├── baseline/
│   ├── folds.json
│   ├── vessel_pretrain/shared/checkpoint_best.pth
│   ├── folds/fold_0..4/checkpoint_best.pth
│   └── ensemble/evaluation/test/metrics.json
├── tensorboard/baseline/
│   ├── vessel_pretrain/shared/
│   └── folds/fold_0..4/
└── pipeline.log
```

### Run과 logging 구조

```text
RUN_DIR/
├── cache/
│   ├── index.json
│   └── cases/<case_id>/{image,location,vessel,instances}.npy
├── cache_report.json
├── baseline/
│   ├── config.json
│   ├── environment.json
│   ├── inputs.json
│   ├── provenance.json
│   ├── model.json
│   ├── status.json
│   ├── training_log.txt
│   ├── checkpoint_latest.pth
│   ├── checkpoint_best.pth
│   ├── checkpoint_best_task1.pth
│   ├── checkpoint_best_task2.pth
│   └── metrics/{train,test,official_val}/epoch_XXXX.json
├── predictions/
└── tensorboard/baseline/
```

Aneurysm 학습은 `loss/train`, train loss component와 learning rate를 기록합니다. Patch
`loss/val`은 checkpoint quality의 proxy로 사용하지 않으며 계산과 logging에서 제거했습니다.
Validation 주기마다 EMA weight로 전체 validation volume을 추론한 뒤 원본
mask grid에서 공식 Task 1/Task 2 metric을 계산합니다. TensorBoard에는
`metric/val`, `official/val/task1/*`, `official/val/task2/*`와
`official/val/checkpoint_selection`이 기록됩니다. `metric/val`은
`official/val/checkpoint_selection`과 같은 mean metric입니다. Vessel pretraining만은
aneurysm 공식 metric이 없으므로 `loss/train`과 `loss/val`을 사용합니다.

Grand Challenge 최종 점수는 여러 제출의 metric별 rank 평균이므로 단일 학습 run에서는 직접
계산할 수 없습니다. 따라서 checkpoint 선택에는 공식 metric의 방향 보정 평균을 사용합니다.
Task 1은 `mean(precision, recall, MCC)`, Task 2는
`mean(precision, recall, MCC, Dice, volumetric similarity, 1-HD95)`입니다.
`validation.selection_task`의 기본값은 `task2`이며 그 최적점이
`checkpoint_best.pth`가 됩니다. Task별 최적점도 `checkpoint_best_task1.pth`와
`checkpoint_best_task2.pth`에 별도로 저장합니다. `checkpoint_latest.pth`에는
model/EMA/optimizer/scheduler/GradScaler/RNG state가 모두 들어가며 validation 주기 및 마지막
epoch에만 원자적으로 갱신합니다.

```bash
tensorboard --logdir "$RUN_DIR/tensorboard/baseline"
```

중단된 run은 같은 위치에서 이어갑니다.

```bash
python scripts/train.py \
  --config configs/baseline.yaml \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --resume "$RUN_DIR/baseline/checkpoint_latest.pth"
```

Train CLI 전체 계약:

| argument | 필수 | 의미 |
|---|---:|---|
| `--config PATH` | 아니오 | 기본값 `configs/baseline.yaml` |
| `--run-dir PATH` | 권장 | `RUN_DIR/cache`, `RUN_DIR/baseline`, `RUN_DIR/tensorboard/baseline` 사용 |
| `--cache PATH` | 조건부 | cache 경로 override; `--run-dir`이 없으면 필수 |
| `--resume PATH` | 아니오 | 같은 run의 `checkpoint_latest.pth` |
| `--device cuda\|cpu\|auto` | 아니오 | YAML의 device override |
| `--output-root PATH` | legacy | name/timestamp 출력 방식; `--run-dir`와 동시 사용 불가 |
| `--smoke-test` | 검사 전용 | 1 epoch, split별 2 patch, worker 0으로 축소 |

Train patch는 lesion component 단위로 sampling하며 희귀 class와 작은 병변을 oversample합니다.
Negative patch의 일부는 vessel point를 중심으로 뽑아 혈관 주변 false positive를 학습합니다.

## 공식 TopAneu-26 방식 평가

`evaluate.py`는 [TopAneu-26 공식 evaluator](https://github.com/Bangulli/TopAneu-26/tree/main/eval)의
계산식을 포트합니다. Task 1은 52개 위치별 Precision/Recall/MCC를 누적한 뒤 macro 평균하고, Task 2는
case-location overlap 기반 Precision/Recall/MCC와 DSC, normalized HD95, Volumetric Similarity를 공식
방식으로 집계합니다. Cache-grid prediction은 nearest-neighbor로 원본 NIfTI grid에 복원한 뒤 원본
location mask와 비교합니다.

```bash
python scripts/evaluate.py \
  --run-dir "$RUN_DIR" \
  --split val \
  --device cuda \
  --no-save-predictions \
  --overwrite
```

원본 데이터 위치는 cache index의 `source_root`를 사용합니다. 데이터가 이동했다면
`--source /new/path/to/topaneu_release`로 지정합니다. 공식 macro 결과는 `metrics.json`의
`official_task1.macro`와 `official_task2.macro`에 저장됩니다. `diagnostics.task2_binary_voxel`은
threshold 분석을 위한 비공식 보조 지표이며 challenge ranking에는 사용되지 않습니다.
`per_case_metrics.json`에는 `task1_location_scores` 52개와
`aneurysm_presence_score`도 저장되므로 prediction volume을 저장하지 않아도 Task 1 threshold를
사후 분석할 수 있습니다.

Validation 추론을 한 번 완료한 뒤 Task 1 threshold는 GPU 재추론 없이 탐색할 수 있습니다.

```bash
python scripts/tune_presence_threshold.py \
  --per-case "$RUN_DIR/baseline/ensemble/evaluation/val/per_case_metrics.json"
```

선택 결과와 전체 sweep은 같은 폴더의 `presence_threshold_sweep.json`에 저장됩니다. Test label로
threshold를 고르면 leakage이므로 반드시 validation 또는 OOF 결과에만 사용합니다.

## 단일 volume 추론

```bash
python scripts/predict.py \
  --image /path/to/case_0000.nii.gz \
  --run-dir "$RUN_DIR" \
  --device cuda
```

출력은 다음과 같습니다.

```text
$RUN_DIR/predictions/
├── task1_locations/<case>.json
└── task2_masks/<case>.nii.gz
```

Predict CLI 전체 계약:

| argument | 필수 | 의미 |
|---|---:|---|
| `--image PATH` | 예 | 입력 NIfTI (`.nii` 또는 `.nii.gz`) |
| `--run-dir PATH` | 권장 | best checkpoint와 prediction 출력 경로를 자동 선택 |
| `--checkpoint PATH` | 조건부 | 기본값 `RUN_DIR/baseline/checkpoint_best.pth` |
| `--output PATH` | 조건부 | 기본값 `RUN_DIR/predictions` |
| `--modality ct\|mr` | 아니오 | 미지정 시 파일명에서 추론 |
| `--device DEVICE` | 아니오 | 기본값 `cuda` |
| `--overlap FLOAT` | 아니오 | 기본값 `0.5` |
| `--mask-threshold FLOAT` | 아니오 | 기본값 `0.45` |
| `--class-threshold FLOAT` | 아니오 | 이전 checkpoint 호환 인자 |
| `--presence-threshold FLOAT` | 아니오 | Task 1 component score 기준, 기본값 `0.35` |
| `--presence-top-k INT` | 아니오 | Task 1 class별 strongest patch 평균 개수, 기본값 `3` |
| `--presence-evidence-voxels INT` | 아니오 | patch gate에 쓰는 strongest voxel 개수, 기본값 `64` |
| `--minimum-component-voxels INT` | 아니오 | 제거할 최소 component 크기, 기본값 `5` |
| `--maximum-components INT` | 아니오 | scan당 유지할 최대 후보 수, 기본값 `5` |

`--run-dir` 없이 추론할 때는 `--checkpoint`와 `--output`을 모두 지정해야 합니다.

추론은 53채널 full-volume accumulator를 만들지 않고 confidence-weighted overlap consensus만
보존하므로 메모리 사용량이 volume 크기에 대해 작습니다. Threshold는 validation에서 조정해야 하며,
Grand Challenge의 `.mha` socket/container adapter는 연구 baseline 성능이 확인된 뒤 붙이는 범위로
남겨 두었습니다.

## 권장 비교 순서

1. 기존 nnU-Net baseline과 동일 split에서 validation 공식 metric 측정
2. `loss.vessel=0`과 `0.1` 비교로 anatomy branch의 순수 기여 확인
3. binary aneurysm head와 location head의 threshold sweep
4. 이득이 확인되면 coarse ROI locator와 model-driven hard-negative mining 순으로 확장

이 구현은 학습 가능한 baseline 코드이며 pretrained weight나 성능 수치는 포함하지 않습니다.
