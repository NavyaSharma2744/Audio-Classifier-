# Sounds of Home — audio in, class probabilities out

Audio goes in one end, a probability per sound class comes out the other.

```
src/features.py   audio    ->  mel numbers     the only file that opens a WAV
src/dataset.py    corpus   ->  X, y, day       builds and caches the rows
src/train.py      X, y     ->  model           selects, scores, ships
src/predict.py    audio    ->  probabilities   what you actually run
src/cnn.py        X, y     ->  CNN             the same job, convolutionally
src/table.py      aligned columns, so the rest can be about their subject
```

**X is 128 numbers per row, computed from audio alone**: 64 mel-band means (what it sounds like) and
64 mel-band standard deviations (whether it holds still). **y is 7 booleans**, read from the PANNs
JSON: is each class present at ≥ 0.2. **day** is carried alongside and used only for splitting,
never as a feature. The cache keeps the three in separate arrays, so a label cannot reach the model
by accident — the strongest correlation between any single X column and any label is 0.62.

`python src/dataset.py csv` exports the whole thing as `data/training_set.csv` (48 MB, 46,637 × 136).
Careful there: in the flat file X and y sit side by side, columns `0:128` and `128:135`. Feed all
136 columns to a model and you have handed it the answers.

The mel settings in `features.py` are copied from the dataset's own metadata, so the model sees the
representation PANNs saw. Audio is loaded with the stdlib `wave` module rather than `librosa.load`,
which would resample to 22 kHz and convert to float by default — we need exactly 48 kHz and exact
integer zeros to spot the erased blocks.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # torch only needed for src/cnn.py
```

### Getting the data

**This repository contains no audio.** The corpus is 9.4 GB and covers people's homes; `data/` is
gitignored entirely. Download [The Sounds of Home](https://www.cvssp.org/data/ai4s/sounds_of_home/)
and place one recorder's files so the tree reads:

```
data/VITALISE data light 2.3/
├── Audio/14/20231116_100000_modified.WAV        28 files
└── Annotations/14/20231116_100000_light.json    28 files
```

Set `RECORDER` in `src/dataset.py` if you use a different one. Everything under `data/` after that
— caches, the fitted model, the CSV export — is generated and disposable.

## Run it

```bash
python src/features.py                            # self-test the audio maths
PYTHONPATH=src python src/dataset.py               # build the rows (reads all 9 GB, a few minutes)
PYTHONPATH=src python src/dataset.py describe      # what ended up in them
PYTHONPATH=src python src/dataset.py csv           # optional flat export
PYTHONPATH=src python src/train.py                 # select, score, ship
PYTHONPATH=src python src/predict.py <file.WAV>    # predict on any recording
```

`dataset.py` builds `data/training_set.npz` once and every later run reuses it. The fitted model
lands in `data/tagger.joblib`, predictions in `reports/`.

## What the data is, and the two catches

A 25% slice (28 files, 28.0 h, 9.4 GB) of [The Sounds of Home](https://arxiv.org/html/2409.11262):
one AudioMoth in one Belgian home, 16–22 Nov 2023, 48 kHz mono.

**The labels are not ground truth.** They are PANNs `Cnn14_DecisionLevelAtt` (2019, 0.43 mAP)
guessing the top 7 AudioSet classes every 0.213 s. No human labelled any of this. So this model
learns *PANNs' judgement*, and every score below is agreement with PANNs, not accuracy. Its loud
common classes are dependable; its rare ones are not — it reports *Snake* in 5.1% of frames.

**The audio is censored.** `_modified` means speech was zeroed for GDPR using the same PANNs model
at a 0.2 threshold. 27% of these 28 hours is digital silence, up to 55% in busy evening files.
Hence two design decisions: erased blocks are excluded everywhere, and **Speech is not a target** —
half its frames *are* the erasure, so training on it teaches the model that silence means speech.

## What the training set contains

`python src/dataset.py describe`

```
one row = 213 ms of audio -> 128 numbers + 7 true/false labels

  recorded              471,828 blocks   28.0 h  100%
  survived erasure      233,185 blocks   13.8 h   49%    <- half the recording is gone
  kept (every 5th)       46,637 blocks    2.8 h   10%

  day           blocks Televis   Music Silence     Dog     Cat    Door Vehicle
  20231116       2,300      49       0     768      61      32     121      56
  20231117      11,506      19       0   7,652     126      39     232      66
  20231118       3,174       0       0   1,637      35      20     117       7
  20231119       2,885       2       0   1,012      58      10      88      68
  20231120       5,710      57       0   3,850      78      38     114      57
  20231121       6,201       4       0   4,557      75      69      97      12
  20231122      14,861      43       0   8,213     207      78     343   1,943
  TOTAL         46,637     174       0  27,689     640     286   1,112   2,209
```

Three things to hold on to. **Half the recording was erased** — 28 hours in, 13.8 out. **Silence is
59% of all rows**; everything else is scraps. And **88% of Vehicle falls on 22 November** — the one
class that looked promising is, more than anything, one afternoon of traffic — and with the fixed
split it drops out entirely, because 21 November has only 12 vehicle blocks to validate on.

## How well it works

A fixed three-way split by day, chronological so the choice is not cherry-picked. Train fits four
candidate models, validation picks which one and where to put its cutoff, test is read once.

```
split         rows                                          days
train       25,575  20231116 20231117 20231118 20231119 20231120
validation   6,201                                      20231121
test        14,861                                      20231122
```

| class | chosen model | valid AP | chance | test AP | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| Silence | boosting deep | 0.965 | 0.553 | 0.978 | 0.92 | 0.95 | **0.94** |
| Door | forest | 0.251 | 0.023 | 0.137 | 0.23 | 0.24 | 0.24 |
| Cat | forest | 0.046 | 0.005 | 0.048 | 0.03 | 0.28 | 0.05 |
| Dog | forest | 0.102 | 0.014 | 0.082 | 0.02 | 0.29 | 0.04 |

Dropped, too few positives in some split: **Television** (127/4/43), **Music** (0/0/0),
**Vehicle** (254/12/1943). Four of seven classes ship.

`chance` is what a coin weighted to the class frequency scores on test. Average precision asks
whether the model can *rank* the class to the top at any cutoff; precision and recall say what
happens once you commit to one. Both are shown because the second is always harsher — Door has a
6× lift over chance and a precision of 0.23, so three of four blocks it flags are not a door.

### A convolutional net does better on the rare classes

`src/cnn.py` trains one small CNN over the raw 29x64 log-mel spectrogram, all classes at once,
on the same split with the same threshold rule. Test average precision, mean of 4 seeds:

| class | trees | CNN | seed sd |
|---|---|---|---|
| Silence | 0.978 | 0.975 | 0.001 |
| Door | 0.137 | **0.177** | 0.005 |
| Cat | 0.048 | **0.144** | 0.012 |
| Dog | 0.082 | 0.080 | 0.002 |

Door and Cat improve well beyond seed noise. The mechanism is the shared trunk: Cat has only 286
positive blocks in 28 hours, far too few to train a convnet of its own, but the convolutional body
is trained by all 46,637 rows at once and Cat only has to supply the final layer's weights.
Multi-label training is doing the work, not the convolution alone.

Caveat that does not go away: this is one test day. Seed variance is tiny, but *split* variance is
not — Door ranged 0.11 to 0.22 across held-out days when we measured it.

**Only Silence is usable.** Everything else is a detector in name only, CNN included: Door's best
precision is 0.26.

Two things worth reading off this table. Model choice was the only change that ever helped:
gradient boosting, random forest and logistic regression differ by 15–30% on the rare classes,
where three separate attempts at better *features* moved nothing. And Door scored 0.251 on
validation against 0.137 on test — with seven uneven days, a single split is one draw from a wide
spread, so treat any one number here as approximate.

### Television and Music cannot be learned from this dataset

Not "are hard" — cannot. TV and music in a living room come with people talking, the anonymisation
removed everything it heard as speech, and it took the television with it. After erased blocks are
correctly excluded, **Music has zero positive blocks in 28 hours** and Television has 174 out of
46,637. `train.py` requires at least 30 positives in each of train, validation and test, so neither
class is shipped.

This was found late, by an earlier version of this pipeline that used `ERASED_THRESHOLD = 0.99`.
Zero-fraction per block is bimodal — real audio near 0.01, erased audio near 0.99 — so that cutoff
sliced through the erased peak and kept half of it. Television then scored AP 0.450 and F1 0.60,
because the classifier had learned to recognise *digital silence*. The correct threshold is anywhere
in the empty middle; the code uses 0.5. Every class that was not confounded scored **better** after
the fix, which is what removing a distractor looks like.

## Known limits

- **The ceiling is PANNs, and the features are not the bottleneck.** Three separate attempts to
  improve the representation moved nothing: ±2 s of context (+0.03), the full within-block
  spectrogram (worse on transients), and swapping 40 homemade log-spaced bands for a proper 64-band
  mel filterbank at PANNs' own hop size (identical, within noise). A higher score would mean more
  faithful reproduction of a 2019 model, snakes included. Getting past it needs human labels.
- **The output is presence, not intensity.** `predict_proba` gives confidence that a class is
  there. Regressing PANNs' actual score was tried and failed everywhere except Silence — that
  number reflects PANNs' opinion of itself, not a property of the room.
- **Erasure correlates with content.** TV contains speech, so TV-heavy minutes are the most
  censored; the model only ever sees the surviving fragments. `predict.py` prints the erased
  fraction per row so you can see when it is guessing from scraps.
- **One home, seven days, one microphone.** Nothing here generalises to another house.

## attic/

Earlier work, kept because it produced findings this README depends on, and safe to delete:
`check_speech_removal.py` (proved the audio is censored), `dataset_summary.py` and `inspect_DS.py`
(the class distribution), `predict_activity.py` (forecasting — a negative result: 28 hours from one
home is ~28 independent samples, and persistence beats it at every horizon), `predict_probability.py`
(the regression above), plus `cutDB.py`, which made this 25% slice.

Their numbers predate the `ERASED_THRESHOLD` fix below, so treat any score in them as superseded.

Nothing in `attic/` is imported by `src/`; deleting the folder breaks nothing.
