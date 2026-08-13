# Singh & Koundal (2024) TCN reproduction

This directory contains an isolated reproduction of:

> Singh, A. K. & Koundal, D. (2024). A Temporal Convolutional Network for modeling raw 3D sequences and air-writing recognition. *Decision Analytics Journal*, 10, 100373.

The intended model is a causal, dilated residual Temporal Convolutional Network operating on raw trajectory sequences. Model, preprocessing, training, and evaluation code will be added in later phases.

## Directory layout

```text
TCN_Singh2024/
├── configs/       # Reproduction configuration files
├── data/          # Local datasets; not committed
├── results/       # Local outputs; not committed
├── src/           # Implementation modules
├── tests/         # Unit and smoke tests
└── reproduction_notes.md
```

## Installation, data, training, and tests

Commands will be documented once their corresponding implementation phases are complete. Use the `newconda` environment for project commands.

## Reproduction status

Repository scaffolding is complete. Architecture reproduction and numerical paper results have not yet been completed.

## Fidelity note

Only settings explicitly supported by the paper will be described as paper-specified. Engineering choices for information the paper does not specify will be documented as reproduction assumptions in [`reproduction_notes.md`](reproduction_notes.md).
