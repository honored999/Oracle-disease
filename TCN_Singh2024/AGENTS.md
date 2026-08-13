# Singh & Koundal 2024 TCN Reproduction Instructions

## Scope

This directory reproduces:

Singh, A. K. & Koundal, D. (2024),
"A Temporal Convolutional Network for modeling raw 3D sequences and
air-writing recognition",
Decision Analytics Journal 10, 100373.

Only work on this TCN reproduction.

Do not modify other model directories.

## Paper-defined architecture

Input:
- raw 3D trajectory sequence
- each time step contains x, y, z coordinates

Preprocessing:
- root-point translation
- Min-Max normalization

TCN:
- 1D causal convolution
- dilated convolution
- residual connection
- ReLU
- Dropout
- 1x1 Conv1D projection when residual dimensions differ
- last temporal output used for classification
- Dense/logits classification head

Paper figure uses:
- kernel size = 3

Training settings explicitly given by the paper:
- Adam
- initial learning rate = 1e-3
- batch size = 32
- 10-fold cross-validation

Datasets:
- RTD
- RTC
- 6DMG

## Important paper ambiguity

The architecture figure shows dilation examples:
- 1, 2, 4

The experiment section describes dilation using powers of 2 with i from 1 to 8.

Do not silently resolve this inconsistency.

The selected implementation must be configurable and documented in
`reproduction_notes.md`.

## Not explicitly specified by the paper

The following must be treated as reproduction assumptions unless stronger
evidence is found:

- hidden/filter channel count
- dropout probability
- epoch count
- scheduler
- weight decay
- random seed
- exact variable-length sequence handling
- exact padding implementation
- complete layer-by-layer channel configuration

Do not hard-code these without documenting them.

## Model implementation requirements

The model must accept:

[B, C, T]

Do not hard-code C = 3 internally.

This allows future mandibular input such as:
- one 3D point: C = 3
- two 3D points: C = 6

Causal convolution must:
- never use future samples;
- preserve temporal sequence length.

Residual block must:
- use identity shortcut when dimensions match;
- use kernel_size=1 Conv1D projection when channel dimensions differ.

Training should use logits + CrossEntropyLoss.
Do not apply Softmax before CrossEntropyLoss.

## Tests

At minimum test:

1. output shape;
2. temporal length preservation;
3. causal behavior;
4. dilation behavior;
5. residual projection;
6. root-point translation;
7. Min-Max normalization;
8. no NaN for constant coordinate dimensions;
9. short forward/backward smoke training.

## Reproduction status

Do not claim the paper result is reproduced until real RTD/RTC/6DMG experiments
have actually been run.

Synthetic data is only for engineering validation.

## Primary source

The local primary paper is expected at:

`TCN_Singh2024/references/Singh_Koundal_2024_TCN.pdf`

When an architectural detail, formula, preprocessing step, dataset description,
or hyperparameter is uncertain, inspect the primary paper before making an
implementation assumption.

Do not silently fill missing paper details from general TCN knowledge.
If the paper does not specify a detail, record it in `reproduction_notes.md`
as a reproduction assumption.