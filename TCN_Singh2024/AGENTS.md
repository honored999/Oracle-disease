# Singh & Koundal 2024 TCN Reproduction Instructions

## Scope

This directory reproduces:

Singh, A. K., & Koundal, D. (2024).
"A Temporal Convolutional Network for modeling raw 3D sequences and
air-writing recognition."
Decision Analytics Journal, 10, 100373.

Only work on this TCN reproduction when tasks are scoped to this directory.

Do not modify other model implementations unless explicitly requested.


## Primary source

The local primary paper is expected at:

`TCN_Singh2024/references/Singh_Koundal_2024_TCN.pdf`

When an architectural detail, formula, preprocessing step, dataset
description, experimental setting, or hyperparameter is uncertain, inspect
the primary paper before making an implementation assumption.

Do not silently fill missing paper details from generic TCN knowledge.

If the paper does not specify a detail, record it in:

`TCN_Singh2024/reproduction_notes.md`

as a reproduction assumption.


## Input representation

The paper operates on raw 3D trajectory sequences.

The internal sequence representation should be:

`[T, C]`

where:

- `T` is temporal length;
- `C` is the number of trajectory variables.

The model interface should use:

`[B, C, T]`

where:

- `B` is batch size.

The implementation must not hard-code `C = 3`.

The original paper uses 3D trajectories, but the architecture should remain
compatible with future multivariate trajectory inputs such as:

- one 3D point: `C = 3`;
- two 3D points: `C = 6`.


## Paper-specified preprocessing

The paper explicitly describes:

- root-point translation;
- Min-Max normalization.

The paper's root translation equation uses the direction:

`root - point`

for each coordinate.

The implementation must preserve the distinction between the literal paper
formula and conventional `point - root` relative coordinates.

Any alternative mode must be explicitly named and must not silently replace
the paper-literal default.

The paper does not sufficiently specify every normalization axis or
constant-channel edge case.

Such implementation choices must remain documented as reproduction
assumptions.


## Paper-defined TCN architecture

The reproduction must implement a one-dimensional Temporal Convolutional
Network using:

- causal convolution;
- dilated convolution;
- residual connections;
- ReLU;
- Dropout.

The paper architecture figure uses:

`kernel_size = 3`

A residual block consists conceptually of:

1. Dilated Causal Conv1D
2. ReLU
3. Dropout
4. Dilated Causal Conv1D
5. ReLU
6. Dropout
7. residual addition

The shortcut should use:

- identity when input and output channels match;
- `Conv1D(kernel_size=1)` projection when dimensions differ.

Causal convolution must not use future samples.

Temporal sequence length must remain unchanged across the causal convolution
and residual block.


## Dilation ambiguity

The paper contains an ambiguity regarding dilation configuration.

The architecture figure illustrates dilation values similar to:

`[1, 2, 4]`

while the experimental text describes powers of two with `i` ranging from
1 to 8.

Do not silently treat one interpretation as unquestionably paper-specified.

Dilation configuration must remain configurable.

The selected default reproduction configuration and its justification must be
recorded in:

`reproduction_notes.md`


## Classification head

The paper states that the last output in the temporal output sequence is used
for classification.

The default reproduction behavior therefore uses:

TCN temporal features
→ last real temporal feature
→ Linear
→ logits

Do not apply Softmax before `CrossEntropyLoss`.

Probabilities may be computed separately during inference or evaluation.


## Variable-length sequences

Real trajectories may have different temporal lengths.

The current internal batching design uses right padding and returns actual
sequence lengths.

The model must use each sample's real final temporal position rather than the
shared padded final position.

Padding must not influence the selected last-real-step classification feature.

Do not remove `lengths` support without explicit justification.


## Paper-specified training settings

The paper explicitly reports:

- optimizer: Adam;
- initial learning rate: `1e-3`;
- batch size: `32`;
- evaluation: `10-fold cross-validation`;
- primary reported metric: accuracy.

These settings may be treated as paper-specified defaults.


## Reproduction assumptions

The paper does not fully specify several implementation details.

Unless stronger primary-source evidence is found, treat items such as the
following as reproduction assumptions:

- hidden/filter channel count;
- Dropout probability;
- number of epochs;
- random seed;
- scheduler;
- weight decay;
- early stopping;
- checkpoint-selection strategy;
- exact variable-length batching behavior;
- exact normalization axis;
- complete layer-by-layer channel configuration;
- exact interpretation of the dilation schedule;
- loss function when not explicitly specified by the paper.

Do not present these as original paper settings.


## Cross-validation

The paper reports 10-fold cross-validation.

The current generic implementation may use sample-level KFold when no subject
or writer grouping is available.

Do not call sample-level KFold:

- subject-independent;
- writer-independent;
- user-independent.

If writer/subject metadata becomes available, preserve it for future grouped
analysis.

Each fold must initialize a fresh:

- model;
- optimizer;
- loss state where applicable.

Do not carry trained weights across folds.


## Dataset locations

Real datasets are shared at the repository level:

- `data/RTD/`
- `data/RTC/`
- `data/6DMG/`

Do not create duplicate real datasets under:

`TCN_Singh2024/data/`

Adapters should convert raw data into the common internal representation:

- `sequence: Tensor[T, C]`
- `label: int`
- optional metadata


## RTD / RTC

Current official-source investigation indicates that RTD and RTC provide
separate serialized feature and label files.

Real-file structure must be verified against locally downloaded official data
before the adapters are described as validated.

Do not claim that a provisional parser is validated solely because it matches
website documentation.

When real files are available, audit:

- serialization format;
- top-level object type;
- sample count;
- feature structure;
- label encoding;
- sequence lengths;
- class distribution;
- malformed samples;
- NaN / Inf;
- writer/subject metadata if present.

Do not automatically merge official training/test files without evidence that
the paper did so.


## 6DMG

6DMG contains more than just spatial position information.

Official data may include fields such as:

- position;
- quaternion;
- acceleration;
- angular velocity.

The Singh & Koundal paper describes using raw 3D trajectories.

Do not assume that "6DMG" means six input channels.

Until stronger evidence is available, feature selection must remain explicit
and configurable.

If position `[x, y, z]` is used as the reproduction default, document this as
the best-supported reproduction assumption rather than an unquestionable
paper fact.

Preserve available subject/writer/session/trial metadata.


## Dataset adapters

Dataset adapters must:

- parse raw data;
- validate raw structure;
- convert to `[T, C]`;
- map labels deterministically;
- preserve useful metadata;
- reuse the existing preprocessing implementation.

Do not duplicate root translation or normalization logic inside individual
dataset adapters.

Do not invent undocumented raw-file formats.


## Data validation

Before running real training, inspect and report at least:

- sample count;
- class count;
- samples per class;
- feature/channel count;
- minimum sequence length;
- maximum sequence length;
- mean sequence length;
- median sequence length;
- empty sequences;
- NaN / Inf;
- malformed samples;
- available writer/subject metadata.

If observed real data conflict with the paper or official documentation, stop
and document the discrepancy before training.


## Testing requirements

Relevant tests should cover:

- causal behavior;
- dilation behavior;
- residual projection;
- temporal length preservation;
- output shape;
- backward pass;
- root translation;
- Min-Max normalization;
- constant channels;
- variable-length collation;
- real-final-step selection;
- dataset parsing;
- label mapping;
- training step;
- evaluation metrics;
- cross-validation split correctness;
- model reinitialization across folds.

Synthetic fixtures are allowed for parser and engineering tests.

Synthetic fixtures and smoke runs are not paper reproduction experiments.


## Reproduction status

Do not claim the paper has been numerically reproduced until actual official
RTD / RTC / 6DMG experiments have been executed and validated.

Architecture implementation, parser implementation, smoke testing, and real
paper-result reproduction are separate milestones.

Keep their status separate in `reproduction_notes.md`.


## Agent execution model

For this reproduction, the main agent acts only as the coordinator,
integrator, validator, and reporter.

The main agent must not directly modify TCN implementation files.

All concrete file modifications must be delegated to scoped subagents.

Recommended specialized roles include:

- paper/data auditor:
  read-only investigation of primary sources and real data;

- model implementer:
  modifies only explicitly assigned model files;

- dataset implementer:
  modifies only dataset/parser files;

- preprocessing implementer:
  modifies only preprocessing-related files;

- training implementer:
  modifies only training/evaluation/CV files;

- test implementer:
  modifies only explicitly assigned test files;

- documentation implementer:
  modifies README, reproduction notes, or reference documentation only;

- reviewer:
  read-only inspection of implementation, tests, paper fidelity, and data
  handling;

- fixer:
  performs narrowly scoped corrections identified by tests or review.

Each assignment must specify:

1. files/directories that may be read;
2. exact files that may be modified;
3. files/directories that must not be modified;
4. expected deliverable;
5. acceptance criteria.

A subagent must not expand its own scope.

Two subagents must not modify the same file concurrently.

If multiple tasks require the same file, execute those tasks sequentially.

The main agent may:

- inspect files;
- inspect diffs;
- run tests;
- run read-only dataset inspection;
- inspect Git status/history;
- coordinate subagents;
- review results;
- commit verified work;
- summarize the phase.

If tests or review identify a problem, the main agent must assign a new scoped
fix task rather than modifying the implementation directly.