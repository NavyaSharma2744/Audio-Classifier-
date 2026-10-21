"""Predict sound tags from audio alone, held out one day at a time.

Average precision is reported against each day's prevalence, which is what a
coin weighted to the class frequency would score. A model that cannot beat
prevalence has learned nothing.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from audio_features import LABEL_THRESHOLD, load, summary_features
from report import write_report

MIN_POSITIVES = 30  # skip a held-out day with too few examples to score


def leave_one_day_out(features, labels, days):
    """Average precision and prevalence per (day, class), averaged over days."""
    scores, prevalences = [], []

    for held_out in np.unique(days):
        test = days == held_out
        day_scores, day_prevalence = [], []

        for column in range(labels.shape[1]):
            target = labels[:, column]
            if target[test].sum() < MIN_POSITIVES or target[~test].sum() < MIN_POSITIVES:
                day_scores.append(np.nan)
                day_prevalence.append(np.nan)
                continue
            model = HistGradientBoostingClassifier(random_state=0)
            model.fit(features[~test], target[~test])
            probability = model.predict_proba(features[test])[:, 1]
            day_scores.append(average_precision_score(target[test], probability))
            day_prevalence.append(target[test].mean())

        scores.append(day_scores)
        prevalences.append(day_prevalence)

    return np.array(scores), np.array(prevalences)


def main():
    data = load()
    features = summary_features(data["spectrograms"])
    labels = data["label_probs"] >= LABEL_THRESHOLD
    days, names = data["day"], data["target_names"]

    print(f"blocks: {len(features):,}   features: {features.shape[1]}   days: {len(np.unique(days))}")
    print(f"\n{'class':<14}{'prevalence':>12}{'chance AP':>11}{'model AP':>10}{'lift':>8}{'days':>6}")

    scores, prevalences = leave_one_day_out(features, labels, days)
    rows = []

    for column, name in enumerate(names):
        model_ap = np.nanmean(scores[:, column])
        chance = np.nanmean(prevalences[:, column])
        scored = int(np.sum(~np.isnan(scores[:, column])))
        if not scored:
            print(f"{name:<14}{'too rare to score':>47}")
            continue
        print(f"{name:<14}{labels[:, column].mean():>11.1%}{chance:>11.3f}"
              f"{model_ap:>10.3f}{model_ap / chance:>7.1f}x{scored:>6}")
        rows.append([name, round(labels[:, column].mean(), 4), round(chance, 4),
                     round(model_ap, 4), round(model_ap / chance, 2), scored])

    write_report("tag_audio", ["class", "prevalence", "chance_ap", "model_ap", "lift", "days"], rows)


if __name__ == "__main__":
    main()
