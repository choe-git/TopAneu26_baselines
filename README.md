# TopAneu26 nnU-Net baseline

Task 1과 Task 2를 하나의 `nnU-Net v2 Residual Encoder M, 3d_fullres` 모델로 해결하는 최종 baseline입니다.

- Task 2: background 0 + location 1..52의 3D multiclass segmentation
- Task 1: Task 2 mask에 존재하는 location label을 JSON integer list로 변환
- split: 환자 그룹 기준 train 283 / validation 65 / test 68
- TensorBoard: train loss는 매 epoch, validation과 monitor-test는 10 epoch마다
- terminal: epoch/train batch/validation 진행률을 tqdm progress bar로 표시
- final test: 학습 종료 후 challenge metric과 Task 1/2 결과 자동 생성
- run folder: `YYYYMMDD_HHMM` 이름으로 자동 생성

## 로컬 D 드라이브 구조

현재 배치는 다음처럼 자동 인식합니다.

```text
topaneu_release/
├── images/
├── location_jsons/
├── location_masks/
├── location_mapping.json
└── baseline_work/                 # 이 저장소
    ├── split.csv                  # 직접 편집 가능
    ├── train.py
    └── runs/                      # 자동 생성, Git 제외
```

서버의 `<repo>/projects/5_TopAneu/topaneu_baseline`, `<repo>/resources/topaneu_release`,
`<repo>/runs/5_TopAneu` 구조도 자동 인식합니다. 다른 배치에서는 `--data-root`, `--workspace`로 지정합니다.

## 설치와 실행

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
PYTHONPATH=$PWD python train.py --train-all --device cuda
```

기본 모델과 실행 설정:

```text
planner       = nnUNetPlannerResEncM
plans         = nnUNetResEncUNetMPlans
configuration = 3d_fullres
trainer       = nnUNetTrainer
eval interval = 10 epochs
```

평가 주기는 `--eval-every N`으로 바꿀 수 있습니다. 중단된 학습은 `--resume`으로 이어서 실행합니다.
새 학습은 `runs/5_TopAneu/YYYYMMDD_HHMM` 폴더를 자동 생성합니다. 같은 분에 두 번 시작하면 `_02`, `_03`을
붙입니다. `--resume`은 가장 최근 run을 자동 선택하며, 특정 run은 `--run-name 20260819_1630`으로 지정합니다.

## split.csv

저장소 루트의 `split.csv`에서 `split` 열만 `train`, `validation`, `test` 중 하나로 변경하면 됩니다.
실행 시 다음을 검증합니다.

- 모든 case의 정확히 한 번 포함
- 동일 환자의 longitudinal scan 분리 금지
- 빈 partition 금지
- 모든 관측 label이 train에 적어도 하나 존재

## TensorBoard

```bash
tensorboard --logdir ../../../runs/5_TopAneu --port 6006
```

주요 tag:

```text
train/loss/total                              # 매 epoch
validation/patch/loss/total                   # 10 epoch마다
validation/patch/dice/mean_foreground         # 10 epoch마다
validation/patch/dice/class_01..52            # 10 epoch마다
monitor_test/periodic/macro/*                  # 10 epoch마다
test/final/macro/*                            # 최종 1회
```

학습 중 반복 평가되는 test는 엄밀한 final test가 아니므로 `monitor_test`로 구분합니다. 최종 challenge metric은
`Precision`, `Recall`, `MCC`, `Dice`, `VolSim`, `HD95`이며 JSON/CSV와 TensorBoard에 함께 저장됩니다.

## 결과

```text
runs/5_TopAneu/
└── YYYYMMDD_HHMM/
    ├── evaluation/{validation,test}/
    ├── predictions/internal_test_outputs/{task1,task2}/
    ├── tensorboard/
    ├── nnUNet_raw/
    ├── nnUNet_preprocessed/
    └── nnUNet_results/
```
