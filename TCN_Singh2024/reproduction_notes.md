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
- No RTD, RTC, or 6DMG file-format parser is implemented because no local data or verified format specification is available. The Dataset is an in-memory adapter for the internal `(sequence [T, C], label int)` contract.
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
- [ ] RTD data available
- [ ] RTC data available
- [ ] 6DMG data available
- [ ] RTD experiment reproduced
- [ ] RTC experiment reproduced
- [ ] 6DMG experiment reproduced
