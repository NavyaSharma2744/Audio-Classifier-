"""Can we forecast household activity 5 minutes ahead? Result: no better than persistence.

Active means the annotations report Speech, Television or Music -- the classes
PANNs is reliable on. Each day is held out in turn, because activity per day
swings from 0% to 94% and any single train/test split is a lottery.

Kept as a recorded negative result: 28 hours from one home is ~28 independent
samples, which is not enough to beat "assume the next 5 minutes look like the
last 2". See tag_audio.py for the task that does have enough data.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from features import load

HISTORY_BINS = 60          # 15 minutes of context
HORIZON_BINS = 20          # predict 5 minutes ahead
RECENT_BINS = 8            # 2 minutes, as a separate summary
STRIDE = 2                 # one sample every 30 seconds
ACTIVE_CLASSES = ["Speech", "Television", "Music"]
ACTIVE_THRESHOLD = 0.2


def build_samples(data):
    probs, vocab = data["probs"], list(data["vocab"])
    recording, date = data["recording"], data["date"]
    columns = [vocab.index(name) for name in ACTIVE_CLASSES]
    active = probs[:, columns].max(axis=1) >= ACTIVE_THRESHOLD

    features, targets, days = [], [], []

    for index in np.unique(recording):
        rows = np.flatnonzero(recording == index)
        for start in range(0, len(rows) - HISTORY_BINS - HORIZON_BINS, STRIDE):
            history = rows[start : start + HISTORY_BINS]
            horizon = rows[start + HISTORY_BINS : start + HISTORY_BINS + HORIZON_BINS]
            features.append(np.concatenate([
                probs[history].mean(axis=0),
                probs[history[-RECENT_BINS:]].mean(axis=0),
                [active[history].mean(), active[history[-RECENT_BINS:]].mean()],
            ]))
            targets.append(active[horizon].mean() > 0.5)
            days.append(date[index])

    return np.array(features), np.array(targets), np.array(days)


def main():
    features, targets, days = build_samples(load())
    print(f"samples: {len(features):,}\n")
    print(f"{'held-out day':<16}{'n':>6}{'active':>9}{'persistence':>13}{'model':>8}")

    persistence_aucs, model_aucs = [], []

    for held_out in np.unique(days):
        test = days == held_out
        if len(set(targets[test])) < 2:
            print(f"{held_out:<16}{test.sum():>6}{targets[test].mean():>8.0%}"
                  f"{'  one class, unscorable':>34}")
            continue

        # Persistence: how active were the last 2 minutes?
        persistence = roc_auc_score(targets[test], features[test][:, -1])
        model = HistGradientBoostingClassifier(random_state=0)
        model.fit(features[~test], targets[~test])
        auc = roc_auc_score(targets[test], model.predict_proba(features[test])[:, 1])

        persistence_aucs.append(persistence)
        model_aucs.append(auc)
        print(f"{held_out:<16}{test.sum():>6}{targets[test].mean():>8.0%}"
              f"{persistence:>13.3f}{auc:>8.3f}")

    wins = sum(m > p for m, p in zip(model_aucs, persistence_aucs))
    print(f"\n{'mean AUC':<16}{'':>6}{'':>8}{np.mean(persistence_aucs):>13.3f}"
          f"{np.mean(model_aucs):>8.3f}")
    print(f"model beats persistence on {wins} of {len(model_aucs)} days")


if __name__ == "__main__":
    main()