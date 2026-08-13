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
- The sign convention for the root-point translation formula will be recorded alongside the eventual implementation after the paper formula is checked directly.

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

## Deviations from the paper

None recorded at repository-scaffolding stage.

## Reproduction status

- [x] architecture implemented
- [ ] preprocessing implemented
- [x] model architecture unit tests passed
- [ ] RTD data available
- [ ] RTC data available
- [ ] 6DMG data available
- [ ] RTD experiment reproduced
- [ ] RTC experiment reproduced
- [ ] 6DMG experiment reproduced
