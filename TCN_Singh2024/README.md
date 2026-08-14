# Singh & Koundal (2024) TCN reproduction

This directory contains an isolated reproduction of:

> Singh, A. K. & Koundal, D. (2024). A Temporal Convolutional Network for modeling raw 3D sequences and air-writing recognition. *Decision Analytics Journal*, 10, 100373.

The implemented architecture is a causal, dilated residual Temporal Convolutional Network operating on raw trajectory sequences. Preprocessing and the in-memory dataset interface are implemented; training and evaluation code will be added in later phases.

## Directory layout

```text
TCN_Singh2024/
├── configs/       # Reproduction configuration files
├── data/          # Local datasets and internal interface notes; data not committed
├── results/       # Local outputs; not committed
├── src/           # Implementation modules
├── tests/         # Unit and smoke tests
└── reproduction_notes.md
```

## Installation, data, training, and tests

Model architecture tests can be run with:

```powershell
conda run -n newconda python -m pytest TCN_Singh2024/tests/test_model.py -v
```

Preprocessing and in-memory trajectory-interface tests can be run with:

```powershell
conda run -n newconda python -m pytest TCN_Singh2024/tests/test_preprocess.py -v
conda run -n newconda python -m pytest TCN_Singh2024/tests/test_dataset.py -v
```

The training and evaluation library accepts verified Dataset objects through `train_one_epoch`, `evaluate`, and `run_cross_validation`. It uses logits with `CrossEntropyLoss`, Adam defaults of `lr=0.001` and batch size `32`, and sample-level 10-fold CV by default. This sample-level split is not subject-independent evaluation.

An engineering-only synthetic smoke run is available:

```powershell
conda run -n newconda python -m TCN_Singh2024.src.train --config TCN_Singh2024/configs/default.yaml --synthetic-smoke --output-dir $env:TEMP\tcn-smoke
```

Synthetic smoke metrics are not paper reproduction results. Architecture and training pipeline are validated with synthetic data; numerical paper results are not yet reproduced. Use the `newconda` environment for project commands.

## Dataset adapters

Adapters read only repository-root paths such as `data/RTD/`, never
`TCN_Singh2024/data/`. Current per-dataset status is:

- RTD: `BLOCKED` by unresolved official dataset-format ambiguity.
- RTC: `VERIFIED` for real-data parsing/representation and TCN forward
  compatibility. This is not numerical paper reproduction.
- 6DMG: real-data validation remains pending.

See the root [`data/README.md`](../data/README.md) before using the adapters.

### RTC real-data validation

The verified RTC files are stored under `data/RTC/raw/` and remain split:

- Main: `features` has shape `(20098, 800)` and `labels` has shape
  `(20098, 26)`.
- Test: `featuresTest` has shape `(5552, 800)` and `labelsTest` has shape
  `(5552, 26)`.

Both label arrays are strict one-hot encodings for 26 classes. Main contains
2 all-zero empty feature samples. The current adapter intentionally rejects
empty sequences; this phase does not choose how a future training phase must
handle those samples.

For non-empty rows, the adapter removes only continuous trailing-zero padding,
requires the remaining scalar length to be divisible by 3, and reshapes the
interleaved XYZ values to `[T, 3]`. Main and Test are loaded separately with no
automatic merge, while `source_index` and split metadata are preserved. The
adapter reuses the existing preprocessing pipeline: paper-literal root
translation followed by per-sample/per-channel Min-Max normalization. It
rejects empty sequences, NaN/Inf, malformed shapes or counts, and invalid
one-hot labels.

The two official filesets contain `20098 + 5552 = 25650` samples. This differs
from public descriptions that may report approximately 30,000 samples; the
dataset-release/count discrepancy remains unresolved pending authoritative
evidence. Singh & Koundal's exact RTC experiment pool also remains unresolved,
so no main-only, Test-only, merged, or predefined-split interpretation is
claimed here. Any future selection must be labeled as paper-specified or as a
reproduction assumption/best-supported choice.

Real RTC parsing and real-data TCN forward compatibility are validated. Formal
training and 10-fold CV have not started, and Singh's reported RTC accuracy has
not been reproduced.

## Reproduction status

Repository scaffolding and architecture implementation are complete. Numerical paper results have not yet been reproduced.

## Fidelity note

Only settings explicitly supported by the paper will be described as paper-specified. Engineering choices for information the paper does not specify will be documented as reproduction assumptions in [`reproduction_notes.md`](reproduction_notes.md).
