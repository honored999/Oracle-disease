# Local data layout

Do not commit real datasets or participant data to this repository.

Place locally obtained datasets under repository-root directories, not under `TCN_Singh2024/data/`:

```text
<repository-root>/data/
├── RTD/          # RTD digit-recognition data
├── RTC/          # RTC character-recognition data
├── 6DMG/         # 6DMG digit/lowercase/uppercase data
└── mandibular/   # Future mandibular trajectory data
```

The `TCN_Singh2024/data/` directory contains only this tracked interface guide; it must not become a second data location.

The source data formats are not yet fixed by this repository. Future loaders will convert each supported source into the internal sample contract:

```text
sequence: [T, C]
label: int
```

For raw 3D trajectories, `C = 3` represents `(x, y, z)`. The model interface will remain configurable for future inputs where `C > 3`.

The current implementation provides only an in-memory Dataset and a batch collate interface; it does not infer or claim an official RTD, RTC, or 6DMG file format. Dataset samples remain unpadded as `sequence [T, C]`. The collate function right-pads time within a batch and returns:

```text
sequences: [B, C, T_max]
lengths: [B]
labels: [B]
```

Pass `lengths` to the classifier so its last-step head selects each sample's real final temporal feature rather than a padding position.

## Phase 5A provisional adapters

RTD and RTC adapters expect author-documented pickle files named `features` and `labels` under `data/RTD/` or `data/RTC/`. They provisionally interpret each feature sample as a flat `x,y,z,...` sequence and each label as one-hot. This has **not** been validated against the real downloaded files.

The 6DMG adapter provisionally expects the documented SQLite layout and, by default, selects only `position_xyz`. It does not use quaternion, acceleration, or angular-speed fields. This is an evidence-bounded reproduction assumption based on Singh & Koundal's “raw 3D trajectories” description, not a confirmed file-level field selection.

Real-file inspection, sample statistics, and integration validation are deferred to Phase 5B; Phase 5A does not run an inspection command against real data.
