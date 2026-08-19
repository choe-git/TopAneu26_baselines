# TopAneu nnU-Net v2 baseline

One 52-class nnU-Net v2 segmentation model is used for both challenge tasks.

- Task 2: its NIfTI output is the vessel-location segmentation.
- Task 1: the non-zero labels present in that segmentation become the required JSON list, for example `[28]`.

There is no separate classifier, manual resampling, intensity normalization, vessel-mask input, augmentation code, or post-processing. nnU-Net performs its own required planning, resampling, normalization, and training augmentation.

## Layout

```text
TopAneu_baseline/
├── split.csv
├── scripts/
│   ├── prepare_dataset.py
│   ├── preprocess.sh
│   ├── train.py
│   ├── predict.py
│   └── locations_from_masks.py
└── requirements.txt
```

The preparation command creates this separate data directory:

```text
topaneu_data/
└── nnUNet_raw/Dataset501_TopAneu/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    ├── labelsTs/
    ├── split.csv
    └── dataset.json
```

`split.csv` is the single source of truth for the split. Train and validation cases are placed in `imagesTr`/`labelsTr`, because nnU-Net preprocesses both before training. Test cases are isolated in `imagesTs`/`labelsTs` and are never used for preprocessing or training. Files are hard-linked when possible and copied only if links are unavailable.

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

## Prepare once

Run the commands from the `TopAneu_baseline` directory:

```bash
python scripts/prepare_dataset.py \
  --source /path/to/topaneu_release \
  --split-csv split.csv \
  --output topaneu_data
```

This step only arranges the original files in nnU-Net format. It does not resample, normalize, or train anything.

## Preprocess once

```bash
bash scripts/preprocess.sh topaneu_data
```

This runs only nnU-Net planning and preprocessing. The generic channel name `angiography` intentionally selects per-case z-score normalization, which is appropriate for a single model trained on both CTA and MRA.

## Train

```bash
python scripts/train.py \
  --data-root topaneu_data \
  --split-csv split.csv
```

`train.py` converts the CSV train/validation rows to nnU-Net's `splits_final.json` and trains fold 0 using exactly that split. The test rows are checked against `imagesTs`/`labelsTs` but are not exposed to training. To resume an interrupted training run, add `--continue-training`.

## Predict both tasks

Input files must use nnU-Net single-channel names such as `case_001_0000.nii.gz`.

```bash
python scripts/predict.py \
  --images topaneu_data/nnUNet_raw/Dataset501_TopAneu/imagesTs \
  --model topaneu_data/nnUNet_results/Dataset501_TopAneu/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --output /path/to/predictions
```

The resulting `task2_masks/` contains multi-class NIfTI masks. `task1_locations/` contains one JSON list per case with the location IDs present in its Task 2 mask. The model argument is the directory ending in `nnUNetTrainer__nnUNetPlans__3d_fullres` and is used directly, so prediction does not depend on a server-specific `nnUNet_results` path.

Grand Challenge supplies one `.mha` image per container call and expects Task 2 at `/output/images/aneurysm-segmentation/output.mha`. That container adapter is deliberately not mixed into this training baseline; the official template should wrap this model after training.
