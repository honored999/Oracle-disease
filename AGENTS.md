# Oracle-disease Repository Instructions

## Repository scope

This repository is used to reproduce and compare multiple trajectory/time-series
classification networks for future mandibular-motion disease classification.

Each paper/model must live in its own top-level directory.

Examples:

- TCN_Singh2024/
- XCM_Fauvel2021/
- MLSTM_FCN_Karim2019/

## Git rules

- The repository root is the only Git repository.
- Never run `git init` inside model subdirectories.
- Do not use force push.
- Do not overwrite existing remote history.
- Do not commit datasets, checkpoints, large outputs, or private subject data.

## Modification boundaries

When working on one model:
- modify only that model directory unless a repository-level file must change;
- do not refactor or modify other model implementations;
- do not create unrelated model implementations.

## Reproduction principles

Always distinguish:
1. settings explicitly stated by the paper;
2. implementation assumptions caused by missing paper details;
3. deviations from the paper.

Never present an assumption as an original paper setting.

## Data

Real datasets must live under `data/` and must not be committed.

Synthetic data may only be used for unit tests and smoke tests.
Synthetic results must never be reported as reproduction results.

## Testing

Each model implementation should include:
- forward-shape tests;
- core architectural behavior tests;
- preprocessing tests;
- a minimal smoke training test.

## Subagents

Subagents may only modify files explicitly assigned to them.
Do not allow two agents to edit the same file concurrently.
The main agent is responsible for integration and review.