# TopAneu nnU-Net v2 baseline

One 52-class nnU-Net v2 segmentation model is used for both challenge tasks.

- Task 2: its NIfTI output is the vessel-location segmentation.
- Task 1: the non-zero labels present in that segmentation become the required JSON list, for example `[28]`.

The network, preprocessing, augmentation, and prediction format remain nnU-Net v2 defaults. Two small training changes add the task-specific inductive bias that the sparse 52-location labels need:

- One loss supervises foreground/background, the five official vessel territories, laterality, and the final 52 locations from the same 53 output logits. There are no extra heads.
- nnU-Net's forced-foreground batch slot chooses a represented training location uniformly, then uses nnU-Net's own foreground crop for a case containing that location.

There is no separate classifier, manual resampling, intensity normalization, vessel-mask input, custom augmentation, or post-processing.

## Layout

```text
TopAneu_baseline/
├── split.csv
├── scripts/
│   ├── prepare_dataset.py
│   ├── prepare_debug_source.py
│   ├── preprocess.sh
│   ├── train.py
│   ├── trainer.py
│   ├── evaluate.py
│   ├── predict.py
│   └── locations_from_masks.py
└── requirements.txt
```

The preparation command creates a separate run directory outside this repository:

```text
DATA_ROOT/
├── nnUNet_raw/Dataset501_TopAneu/
│   ├── imagesTr/
│   ├── labelsTr/
│   ├── imagesTs/
│   ├── labelsTs/
│   ├── split.csv
│   ├── class_cases.json
│   └── dataset.json
├── nnUNet_preprocessed/
├── nnUNet_results/
└── tensorboard/
```

On the server, `DATA_ROOT` is one timestamped experiment directory:

```text
runs/5_TopAneu/baseline/
└── YYYYMMDD_HHMMSS/
    ├── nnUNet_raw/
    ├── nnUNet_preprocessed/
    ├── nnUNet_results/
    └── tensorboard/
```

`split.csv` is the single source of truth for the split. Train and validation cases are placed in `imagesTr`/`labelsTr`, because nnU-Net preprocesses both before training. Test cases are isolated in `imagesTs`/`labelsTs` and are not used for planning, offline preprocessing, or weight updates; nnU-Net preprocesses them only when evaluation inference runs. Files are hard-linked when possible and copied only if links are unavailable.

The split is patient-level and deterministic:

- train: 326 patients, 334 scans
- validation: 41 patients, 41 scans
- test: 41 patients, 41 scans

Longitudinal scans from the same patient always remain in the same split. CTA/MRA, center, positive/negative, and available location-label distributions are balanced as far as the dataset permits. Labels with only one or two positive cases remain in train.

## Install

Use a clean Python environment with a CUDA-compatible PyTorch install, then:

```bash
python -m pip install -r requirements.txt
```

## Server paths

Run every command from:

```bash
cd /home/introai30/.apni/users/yhchoe/projects/5_TopAneu/TopAneu_baseline
```

The paths in the server layout are:

```bash
SOURCE_ROOT=/home/introai30/.apni/users/yhchoe/projects/resources/topaneu_release
RUNS_ROOT=/home/introai30/.apni/users/yhchoe/projects/runs/5_TopAneu/baseline
RUN_ID=$(date +%Y%m%d_%H%M%S)
DATA_ROOT="$RUNS_ROOT/$RUN_ID"
echo "$DATA_ROOT"
```

Create `RUN_ID` only once per experiment and use the same `DATA_ROOT` for preparation, preprocessing, and training. `SOURCE_ROOT` is read-only input. All arranged data, preprocessing files, checkpoints, predictions, metrics, and TensorBoard events are written below the timestamped `DATA_ROOT`.

## Prepare once

Run the commands from the `TopAneu_baseline` directory:

```bash
python scripts/prepare_dataset.py \
  --source "$SOURCE_ROOT" \
  --split-csv split.csv \
  --output "$DATA_ROOT"
```

This step only arranges the original files in nnU-Net format and writes the small `class_cases.json` index used by the training sampler. It does not resample, normalize, or train anything.

- `--source`: original `topaneu_release` directory
- `--output`: timestamped experiment directory; no files are written into the source data
- `--split-csv`: CSV containing exactly one `1` among `train`, `val`, and `test` per case
- `--overwrite`: rebuild only `nnUNet_raw/Dataset501_TopAneu`; normally use a new `DATA_ROOT` instead

## Preprocess once

```bash
bash scripts/preprocess.sh "$DATA_ROOT"
```

This runs only nnU-Net planning and preprocessing. The generic channel name `angiography` intentionally selects per-case z-score normalization, which is appropriate for a single model trained on both CTA and MRA.

## Train

```bash
python scripts/train.py \
  --data-root "$DATA_ROOT" \
  --split-csv split.csv \
  --device cuda
```

`train.py` converts the CSV train/validation rows to nnU-Net's `splits_final.json` and trains using exactly that split. The test rows are checked against `imagesTs`/`labelsTs` and are used only for evaluation. To resume an interrupted training run, add `--continue-training`.

- `--data-root`: the same timestamped `DATA_ROOT` used for preparation and preprocessing
- `--split-csv`: split definition; defaults to the repository's `split.csv`
- `--continue-training`: resume `checkpoint_latest.pth`
- `--device`: `cuda` for the server, or `cpu` for diagnostics only
- `--smoke-test`: 10 epochs with one train/validation iteration; never use for the full experiment
- `--epochs`, `--train-iterations`, `--val-iterations`: optional debug-budget overrides

Training uses the nnU-Net v2 defaults for 1,000 epochs and 250 training iterations per epoch. Train loss is written every epoch. Every 10 epochs, the trainer computes validation loss and runs full-volume inference on the validation and test sets. TensorBoard records:

- `loss/train` every epoch
- `loss/val` every 10 epochs
- `val/task1/*`, `val/task2/*`, `test/task1/*`, and `test/task2/*` every 10 epochs

Task 1 metrics are Precision, Recall, and MCC. Task 2 metrics are Precision, Recall, MCC, Dice, volumetric similarity, and normalized HD95, using the official 52-class TopAneu aggregation ([Task 1 evaluator](https://github.com/Bangulli/TopAneu-26/blob/main/eval/task1/evaluate.py), [Task 2 evaluator](https://github.com/Bangulli/TopAneu-26/blob/main/eval/task2/evaluate.py)). Per-class values are saved under `evaluation/metrics/epoch_XXXX/`; TensorBoard and the terminal show the class averages. The latest validation and test masks are overwritten in fixed folders, and the final test averages are printed when training ends.

To view the logs:

```bash
tensorboard --logdir "$DATA_ROOT/tensorboard"
```

## Required checks before the full run

The helper creates only hard links when possible. It never edits the release data.

### 1. Positive one-case overfit

This aliases one positive location-49 case into train/validation/test. Leakage is intentional: the only question is whether the complete pipeline can learn one real scan. Its metrics must not be reported as model performance.

```bash
DEBUG_SOURCE=/tmp/topaneu_overfit
OVERFIT_ROOT="$RUNS_ROOT/$(date +%Y%m%d_%H%M%S)"

python scripts/prepare_debug_source.py \
  --source "$SOURCE_ROOT" \
  --output "$DEBUG_SOURCE" \
  --profile overfit \
  --overwrite
python scripts/prepare_dataset.py \
  --source "$DEBUG_SOURCE" \
  --split-csv "$DEBUG_SOURCE/split.csv" \
  --output "$OVERFIT_ROOT"
bash scripts/preprocess.sh "$OVERFIT_ROOT"
python scripts/train.py \
  --data-root "$OVERFIT_ROOT" \
  --split-csv "$DEBUG_SOURCE/split.csv" \
  --smoke-test \
  --epochs 50 \
  --train-iterations 20 \
  --device cuda
```

Pass criterion: loss clearly falls and the model produces foreground for the repeated case. The exact epoch is not a challenge score.

### 2. Eight-case mini test

This uses four train, two validation, and two test scans. The fixed cases include a common location, one-shot/rare locations, a multi-location case, negative scans, CTA, and MRA.

```bash
DEBUG_SOURCE=/tmp/topaneu_mini
MINI_ROOT="$RUNS_ROOT/$(date +%Y%m%d_%H%M%S)"

python scripts/prepare_debug_source.py \
  --source "$SOURCE_ROOT" \
  --output "$DEBUG_SOURCE" \
  --profile mini \
  --overwrite
python scripts/prepare_dataset.py \
  --source "$DEBUG_SOURCE" \
  --split-csv "$DEBUG_SOURCE/split.csv" \
  --output "$MINI_ROOT"
bash scripts/preprocess.sh "$MINI_ROOT"
python scripts/train.py \
  --data-root "$MINI_ROOT" \
  --split-csv "$DEBUG_SOURCE/split.csv" \
  --smoke-test \
  --epochs 20 \
  --train-iterations 20 \
  --device cuda
```

### 3. Full-data fresh training

After both checks pass, create a new timestamp and run the normal prepare, preprocess, and train commands above. Do not add `--continue-training`: the previous all-background checkpoint was optimized with a different loss and is not a valid starting point for this baseline.

Use `--device cpu` only for diagnostics; 3D training is much slower on CPU.

## Predict both tasks

Input files must use nnU-Net single-channel names such as `case_001_0000.nii.gz`.

```bash
python scripts/predict.py \
  --images "$DATA_ROOT/nnUNet_raw/Dataset501_TopAneu/imagesTs" \
  --model "$DATA_ROOT/nnUNet_results/Dataset501_TopAneu/nnUNetTrainer__nnUNetPlans__3d_fullres" \
  --output /path/to/predictions
```

The resulting `task2_masks/` contains multi-class NIfTI masks. `task1_locations/` contains one JSON list per case with the location IDs present in its Task 2 mask. The model argument is the directory ending in `nnUNetTrainer__nnUNetPlans__3d_fullres` and is used directly, so prediction does not depend on a server-specific `nnUNet_results` path.

Grand Challenge supplies one `.mha` image per container call and expects Task 2 at `/output/images/aneurysm-segmentation/output.mha`. That container adapter is deliberately not mixed into this training baseline; the official template should wrap this model after training.
