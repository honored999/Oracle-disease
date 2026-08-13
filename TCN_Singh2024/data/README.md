# Local data layout

Do not commit real datasets or participant data to this repository.

Place locally obtained datasets under the following directories:

```text
data/
├── RTD/          # RTD digit-recognition data
├── RTC/          # RTC character-recognition data
├── 6DMG/         # 6DMG digit/lowercase/uppercase data
└── mandibular/   # Future mandibular trajectory data
```

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
