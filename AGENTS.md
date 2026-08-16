# Oracle-disease Repository Instructions

## Repository scope

This repository is used to reproduce, validate, and compare multiple
trajectory and multivariate time-series classification networks.

The long-term goal is to evaluate candidate architectures for mandibular
motion trajectory classification and disease classification.

Each paper or model reproduction must live in its own top-level directory.

Examples:

- `TCN_Singh2024/`
- `XCM_Fauvel2021/`
- `MLSTM_FCN_Karim2019/`

Do not mix unrelated model implementations into the same model directory.


## Repository structure

The repository root is the shared project root.

Real datasets are shared across models and must live under:

`data/`

Do not create duplicate copies such as:

- `TCN_Singh2024/data/`
- `XCM_Fauvel2021/data/`

unless a future task explicitly requires a small model-specific test fixture.

Model-specific source code, configuration, tests, documentation, and
reproduction notes belong inside the corresponding model directory.


## Git rules

The repository root is the only Git repository.

Never run `git init` inside model subdirectories.

Do not:

- create nested Git repositories;
- use `git push --force`;
- rewrite remote history without explicit user approval;
- delete or overwrite unrelated user files;
- commit real datasets;
- commit subject/patient data;
- commit large checkpoints or generated experiment artifacts.

Before committing:

- inspect `git status`;
- inspect relevant `git diff`;
- verify that real data files are ignored.

Use clear, scoped commit messages.

Examples:

- `feat(tcn): implement causal dilated residual network`
- `test(tcn): add architecture behavior tests`
- `docs(repo): update dataset management rules`


## Modification boundaries

When assigned to one model, primarily modify only that model directory.

Repository-level files may be changed only when the change is genuinely
shared across the whole repository, for example:

- `.gitignore`
- root `AGENTS.md`
- root `README.md`
- `data/README.md`

Do not refactor or modify another model implementation unless explicitly
requested.

Do not create implementations for future models before they are assigned.


## Reproduction principles

For every paper reproduction, always distinguish between:

1. settings explicitly stated by the paper;
2. ambiguities or missing details in the paper;
3. reproduction assumptions introduced by this implementation;
4. deliberate deviations from the paper.

Never present a reproduction assumption as an original paper setting.

When a detail is uncertain:

1. inspect the primary paper;
2. inspect the original dataset or official dataset documentation if relevant;
3. inspect cited primary sources if needed;
4. only then introduce a documented reproduction assumption.

Do not silently replace an unclear paper detail with standard deep-learning
practice.

All unresolved or assumed details should be recorded in the corresponding:

`reproduction_notes.md`


## Primary sources

Local primary-paper PDFs may be stored under:

`<model-directory>/references/`

Paper PDFs are local reference material and should normally remain ignored by
Git.

Tracked reference metadata should be stored in:

`<model-directory>/references/README.md`

When reproducing a paper, primary sources take precedence over memory or
generic knowledge of the architecture.


## Data

All real datasets must live under the repository-level:

`data/`

Examples:

- `data/RTD/`
- `data/RTC/`
- `data/6DMG/`
- `data/mandibular/`

Real datasets must not be committed.

Do not move, rewrite, convert, rename, or delete original dataset files unless
explicitly required.

When possible, preserve an untouched raw copy of downloaded data.

Derived or preprocessed data should be placed in clearly separated generated
directories and must not overwrite the raw data.

Synthetic data may be used only for:

- unit tests;
- parser fixtures;
- smoke tests;
- engineering validation.

Synthetic-data results must never be reported as paper reproduction results.


## Generated artifacts

Do not place large generated files or experiment intermediates directly in
source-code directories.

Generated artifacts such as:

- `.npy`
- `.npz`
- predictions;
- cached or preprocessed arrays;
- temporary experiment outputs;
- checkpoints;
- model exports;
- training logs;

should be stored under dedicated generated-output directories such as:

- `outputs/`
- `runs/`
- `checkpoints/`
- `logs/`
- `results/generated/`

Do not add a repository-wide `*.npy` or `*.npz` ignore rule solely to hide
poorly organized generated files.

Small test fixtures may intentionally use `.npy` or `.npz` and may be tracked
when they are necessary for reproducible tests.


## Results

Generated experiment outputs should not be committed by default.

If final validated metrics need to be version controlled, keep them small and
place them in a clearly separated location such as:

`results/reported/`

Do not commit large raw prediction arrays, checkpoints, per-epoch caches, or
temporary smoke-test results.

Synthetic smoke-test metrics must never be presented as real experimental
results.


## Data leakage

Avoid train/validation/test leakage.

Preprocessing that depends on dataset-level statistics must be fitted only on
the appropriate training split unless the original paper explicitly specifies
otherwise.

If subject or writer identifiers exist, preserve them as metadata.

Do not describe sample-level cross-validation as subject-independent
evaluation.


## Testing

Each model implementation should include tests appropriate to its core
architecture and data pipeline.

Where applicable, test:

- forward shape;
- numerical finiteness;
- core architectural behavior;
- preprocessing;
- variable-length sequences;
- dataset parsing;
- training step;
- evaluation metrics;
- data split correctness;
- end-to-end smoke execution.

Tests should verify behavior, not merely that code executes.


## Configuration

Important experimental parameters should be configuration-driven rather than
scattered as hard-coded constants.

Paper-defined settings and reproduction assumptions must remain distinguishable
in both configuration and documentation.

## Main-agent orchestration policy

The main agent should act primarily as the project coordinator, planner,
integrator, and final reviewer.

By default, the main agent should not directly edit implementation files.

Concrete file modifications should be delegated to clearly scoped subagents.

This includes modifications to:

- source code;
- tests;
- configuration files;
- dataset adapters;
- preprocessing code;
- training code;
- evaluation code;
- model-specific documentation;
- reproduction notes.

The main agent is responsible for:

- reading repository instructions and current project state;
- inspecting Git status and diffs;
- decomposing the task into independent work units;
- assigning each work unit to a subagent;
- defining exact read/write boundaries for every subagent;
- sequencing dependent tasks;
- reviewing subagent reports and diffs;
- running or coordinating tests;
- identifying integration problems;
- assigning follow-up fix tasks to subagents;
- coordinating independent read-only review;
- deciding whether the phase satisfies its acceptance criteria;
- creating the final Git commit after verification;
- summarizing the completed work for the user.

The main agent may execute read-only or validation commands such as:

- `git status`;
- `git diff`;
- `git log`;
- searches;
- file inspection;
- test commands;
- dataset inspection commands that do not modify raw data.

If an implementation problem is discovered, the main agent should not
silently fix the code itself.

Instead, it should create a narrowly scoped follow-up task for an appropriate
subagent, then review and test the resulting change.

Exceptions are allowed only for trivial emergency changes when delegation is
impossible, and such exceptions must be explicitly reported to the user.

The preferred workflow is:

planning
→ scoped subagent implementation
→ main-agent inspection
→ independent reviewer
→ scoped subagent fixes if required
→ main-agent validation
→ Git commit
→ final summary


## Subagents

Subagents may be used to perform clearly scoped work.

Every subagent assignment must specify:

- files it may read;
- files it may modify;
- files or directories it must not modify;
- expected output;
- acceptance criteria.

Subagents must not modify files outside their assigned scope.

Do not allow two subagents to modify the same file concurrently.

The main agent is responsible for:

- task decomposition;
- integration;
- conflict resolution;
- testing;
- final review;
- Git operations.

Reviewer subagents should normally be read-only unless explicitly authorized
to modify files.


## Safety and scope

Do not:

- download data from untrusted mirrors when an official source exists;
- bypass dataset access restrictions;
- expose private subject data;
- modify system-level Python environments without explicit need;
- install large dependency stacks without checking necessity;
- claim numerical reproduction before real experiments have actually run.

When blocked by missing data, unavailable services, or ambiguous paper
information, report the blocker explicitly rather than inventing a solution.