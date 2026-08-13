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

No implementation assumptions have been selected yet. Later assumptions will be configurable and explicitly recorded here before being described as operational defaults.

## Deviations from the paper

None recorded at repository-scaffolding stage.

## Reproduction status

- [ ] architecture implemented
- [ ] preprocessing implemented
- [ ] unit tests passed
- [ ] RTD data available
- [ ] RTC data available
- [ ] 6DMG data available
- [ ] RTD experiment reproduced
- [ ] RTC experiment reproduced
- [ ] 6DMG experiment reproduced
