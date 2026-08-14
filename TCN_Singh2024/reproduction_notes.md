# Reproduction notes: Singh & Koundal (2024) TCN

## Paper-specified settings

- Task: air-writing classification from raw 3D trajectory sequences.
- Raw trajectory coordinates: `(x, y, z)`.
- Preprocessing: root-point translation and Min-Max normalization.
- TCN residual block: two dilated causal 1D convolutions, each followed by ReLU and dropout, with an identity or 1x1 convolution residual path.
- Kernel/filter size shown in the architecture figure: 3.
- Dilation values shown in the architecture figure: `[1, 2, 4]`.
- Classifier: the final temporal TCN output is used for classification.
- Optimizer: Adam.
- Initial learning rate: `0.001`.
- Batch size: `32`.
- Evaluation: 10-fold cross-validation.
- Datasets discussed: RTD (10-class digit recognition), RTC (26-class character recognition), and 6DMG (digits, lowercase, uppercase).

## Ambiguities in the paper

- The architecture figure shows dilations `[1, 2, 4]`, while the experiment text describes multiple levels using `2^i` for `i = 1 ... 8`.
- The number of filters/hidden channels per layer is not explicitly specified.
- Dropout probability is not explicitly specified.
- Number of epochs, scheduler, weight decay, early stopping, and random seed are not explicitly specified.
- Variable-length trajectory handling, including padding and truncation policy, is not specified in enough detail.
- The paper does not fully specify the complete channel layout for every layer.
- Root-point translation Eq. (2) writes the transformed coordinate with the same symbol as the input, in the direction `x_0 - x_i` (and equivalently for `y` and `z`). The notation is potentially confusing because it does not distinguish the transformed variable with a prime.

## Reproduction assumptions

- Default dilation list: `[1, 2, 4]`. This follows the architecture figure rather than resolving the conflicting `2^i, i = 1 ... 8` experiment-text description. The list is configurable in YAML and model construction.
- Default hidden-channel layout: `[32, 32, 32]`. The paper does not specify filter counts; this is an engineering default, not a paper setting.
- Default dropout probability: `0.2`. The paper requires dropout in each residual-block main branch but does not state its probability; this is an engineering default, not a paper setting.
- Causal convolutions use explicit left-only zero padding of `dilation * (kernel_size - 1)` and no right padding. This implementation detail is needed to preserve length and causality but is not fully specified by the paper.
- Both convolutions in a residual block use that block's single configured dilation. The paper does not give an independent dilation per convolution.
- The residual addition has no post-add activation. The paper records ReLU after each main-branch convolution but does not explicitly state a post-add activation.
- All convolutions use PyTorch defaults: `stride=1`, `bias=True`, and PyTorch default parameter initialization. These are framework choices, not paper-specified settings.
- The implementation adds no normalization, pooling, attention, or Softmax layer. The classifier uses the final temporal feature and returns raw logits.
- The architecture accepts any positive temporal length. Variable-length batching, padding/masking, and truncation policies are deferred to the preprocessing/data phase; an empty sequence is rejected with `ValueError`.
- The model configuration is a project-level default for engineering validation only. It is not an RTD, RTC, or 6DMG experiment configuration and must be adapted to verified dataset metadata before a real experiment.
- Root-point translation defaults to the paper-literal direction: `root - point` (`x_0 - x_i`). An explicit `conventional_relative` mode (`point - root`) is provided only as a named engineering alternative; modes are never mixed within one call.
- Min-Max normalization is applied independently to every sample and channel across that sample's temporal axis: for `[T, C]`, each channel uses its own `min(sequence[:, c])` and `max(sequence[:, c])`. The paper gives the scalar formula and target range `(0, 1)`, but not this axis choice; it is a reproduction assumption. This per-sample operation avoids using validation/test samples or cross-sample statistics.
- A constant per-sample channel is mapped to zero using a safe denominator, so normalization never produces NaN or Inf. The paper does not specify constant-channel behavior.
- Each Dataset sample remains an unpadded `[T, C]` tensor with integer label. The batch collate function right-pads time only, returns model-ready `sequences: [B, C, T_max]`, `lengths: [B]`, and `labels: [B]`. This variable-length batching policy is a reproduction assumption; the paper does not specify it.
- When `lengths` are supplied to the classifier, it selects feature index `lengths[b] - 1` for each sample instead of the shared padded end. This minimal model extension prevents padding from replacing a sample's real final time step and preserves the prior behavior when `lengths=None`.
- RTD adapter implementation: provisional; real-file validation: pending. It follows the author page's `features`/`labels` pickle description, flat XYZ sequence order, and one-hot labels, but exact pickle containers, label vocabulary, metadata, and values are unverified until actual files are inspected.
- RTC adapter implementation: the real-file parsing and representation behavior described in the RTC validation findings below is validated. This validation does not establish a numerical paper reproduction or an RTC experiment-pool choice.
- 6DMG adapter implementation: provisional; real-file validation: pending. It follows the documented SQLite record layout and defaults to `position_xyz` only. Singh & Koundal describe raw 3D trajectories while noting that 6DMG also has inertial fields; choosing position XYZ is the current evidence-bounded reproduction assumption, not a claimed explicit column list from their paper.
- The Phase 5A 6DMG adapter assumes an unverified SQLite table with `name`, `tester`, `trial`, `length`, and raw `data` BLOB columns; a little-endian 14-float32 record (`timestamp`, position XYZ, quaternion WXYZ, acceleration XYZ, angular-speed XYZ); and position at record offsets 1-3. These serialization details are provisional loader/database assumptions awaiting Phase 5B real-file and official-loader validation.
- `position_xyz` is the only implemented Phase 5A feature selection. The parameter is explicit so a future verified selection can be added, but quaternion, acceleration, and angular speed are intentionally unsupported until evidence establishes their intended use.
- Adapters apply Phase 3 preprocessing in the order root-point translation then per-sample/per-channel Min-Max normalization. The paper specifies both transforms but does not explicitly specify their strict order; this order is a reproduction assumption. Root translation alone sets the first point to zero; subsequent Min-Max normalization can shift that value.
- RTD and 6DMG real-file validation, real subject/writer metadata, 6DMG table/version, and dataset-specific field validation outside the RTC findings below remain pending. RTC parsing and real-data TCN forward compatibility are validated, but no real-data numerical result or paper experiment is claimed.
- Default training configuration preserves the paper-specified Adam optimizer, initial learning rate `0.001`, batch size `32`, and 10 folds. The default `num_classes=10` is only an RTD-oriented placeholder and must be changed to verified dataset metadata before a real experiment.
- Epochs (`20`), random seed (`42`), DataLoader shuffle, worker count, drop-last behavior, device choice, CrossEntropyLoss, no gradient clipping, no scheduler, no weight decay, no early stopping, and no checkpoint-selection policy are reproduction assumptions or explicit omissions; they are configuration-controlled where applicable.
- Cross-validation uses reproducible shuffled, non-stratified, sample-level KFold with fold seed `seed + fold`. It is not subject-independent evaluation and must not be described as such without verified subject IDs and a grouped split.
- Fold-level validation accuracy is aggregated as arithmetic mean and population standard deviation. Accuracy is the paper-reported main metric; loss, confusion matrix, per-class accuracy, standard deviation, and metrics JSON are engineering supplements. Per-class accuracy is `null` when a class has no validation examples in a fold.
- Training and evaluation pass batch `lengths` to the classifier. Synthetic smoke runs use a two-class, ten-sample, one-epoch CPU dataset solely to validate orchestration; their metrics are not RTD, RTC, 6DMG, or paper reproduction results.

## Deviations from the paper

None recorded at repository-scaffolding stage.

## Reproduction status

- [x] architecture implemented
- [x] preprocessing implemented
- [x] model architecture unit tests passed
- [x] preprocessing and dataset-interface unit tests passed
- [x] training, evaluation, and sample-level CV pipeline validated on synthetic data
- [ ] RTD adapter real-file validated (provisional implementation exists)
- [x] RTC adapter real-file validated for parsing/representation and TCN forward compatibility; this is not numerical reproduction
- [ ] 6DMG adapter real-file validated (provisional implementation exists)
- [ ] RTD data available
- [x] RTC data available
- [ ] 6DMG data available
- [ ] RTD experiment reproduced
- [ ] RTC experiment reproduced
- [ ] 6DMG experiment reproduced

## Real dataset validation status

### RTD

- Status: `BLOCKED` by official dataset-format ambiguity.
- Official source identified: yes
- Official files downloaded: not resolved in this phase
- Expected local location: `data/RTD/raw/`
- Raw-file audit: blocked by the official dataset-format ambiguity
- Adapter implementation: provisional / pending
- Real-file adapter validation: pending
- Sample statistics verified: no
- Paper 10-fold data usage confirmed: no
- Paper numerical experiment reproduced: no

Official files may include:

- `features`
- `featuresTest`
- `labels`
- `labelsTest`

The existence of an official test split does not establish whether Singh &
Koundal used only the main set, only a predefined split, or a merged dataset
for their reported 10-fold cross-validation. This remains unresolved until
supported by primary-source evidence.

### RTC

- Status: `VERIFIED` for real-data parsing/representation and TCN forward compatibility.
- Official source identified: yes
- Official files: available under `data/RTC/raw/`
- Expected local location: `data/RTC/raw/`
- Raw-file audit: verified for the facts listed below
- Adapter implementation: validated for the verified real-file representation
- Real-file adapter validation: verified for parsing/representation and TCN forward compatibility
- Sample statistics verified: shape and empty-sample findings below; no class-frequency table is claimed here
- Paper numerical experiment reproduced: no

Verified RTC facts:

- Main: `features` shape `(20098, 800)`; `labels` shape `(20098, 26)`.
- Test: `featuresTest` shape `(5552, 800)`; `labelsTest` shape `(5552, 26)`.
- Both label arrays are strict one-hot encodings for 26 classes.
- Main contains 2 all-zero empty feature samples.
- Non-empty trajectories are recoverable variable-length interleaved XYZ
  sequences. Only continuous trailing-zero padding is removed; the valid
  non-empty scalar length must be divisible by 3, then the values are reshaped
  to `[T, 3]`.
- Main and Test remain separate; the adapter does not automatically merge
  them. `source_index` and split metadata are preserved.
- The adapter reuses the existing preprocessing: paper-literal root
  translation followed by per-sample/per-channel Min-Max normalization.
- The adapter rejects empty sequences, NaN/Inf, malformed shapes or counts,
  and invalid one-hot labels.

The current adapter intentionally rejects the 2 empty Main sequences. Future
training must explicitly decide and document their handling; this validation
does not prescribe dropping them.

The official files contain `20098 + 5552 = 25650` samples. Public descriptions
may report approximately 30,000 samples; this dataset-release/count discrepancy
is unresolved pending authoritative evidence, with no explanation asserted
here.

Singh & Koundal's exact RTC experiment pool remains unresolved. This phase does
not claim main-only, Test-only, merged, or predefined-split usage. Any future
selection must be labeled as paper-specified or as a reproduction
assumption/best-supported choice.

Formal RTC training and 10-fold CV have not started. Singh's reported RTC
accuracy has not been reproduced.

### 6DMG

- Official source identified: yes
- Official files downloaded: pending
- Expected local location: `data/6DMG/raw/`
- Raw-file audit: pending
- Adapter implementation: provisional / pending
- Sample statistics verified: no
- Exact Singh & Koundal input-field selection confirmed: no
- Paper numerical experiment reproduced: no

The paper describes use of raw 3D trajectories. Position `[x, y, z]` is
currently the best-supported candidate input for 6DMG, but the exact field
selection has not yet been established as an explicit paper-specified fact.

## Deliberate validation findings

The RTC observations above are validation findings about the verified real
files and current adapter behavior. They are not additional paper-specified
training settings. In particular, this phase deliberately stops at real-data
parsing and forward compatibility; it does not select an RTC experiment pool,
start formal training, run 10-fold CV, or report a reproduced accuracy.

## Unresolved issues

1. Main contains 2 all-zero empty feature samples, and the current adapter
   intentionally rejects empty sequences. Future training must explicitly
   decide and document their handling; no dropping policy is selected here.
2. The official files contain 25,650 samples in total, while public
   descriptions may report approximately 30,000. The dataset-release/count
   discrepancy remains unresolved pending authoritative evidence.
3. Singh & Koundal's exact RTC experiment pool remains unresolved. Main-only,
   Test-only, merged, and predefined-split usage are not claimed. Future
   selection must be labeled paper-specified or a reproduction
   assumption/best-supported choice.
4. RTD remains `BLOCKED` by official dataset-format ambiguity.
