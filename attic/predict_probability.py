"""How strong is the class, not just whether it is there.

The JSON kept only the top 7 classes per frame, so a missing class means
"somewhere below the 7th score", not zero. Regressing over those frames would
train the model toward a value we know is wrong, so each class is fitted only
on the frames where its score was actually recorded. That makes this a
conditional model: given the class is present, how strong is it?

Pair it with tag_audio.py, which answers whether it is present at all.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from audio_features import load, summary_features
from report import write_report

MIN_FRAMES = 500  # a held-out day needs this many recorded frames to be scored


def correlation(truth, predicted):
    if truth.std() < 1e-6 or predicted.std() < 1e-6:
        return np.nan
    return float(np.corrcoef(truth, predicted)[0, 1])


def score_class(features, target, days, recorded):
    """Leave-one-day-out MAE and correlation, against predicting the training mean."""
    model_errors, baseline_errors, correlations = [], [], []

    for held_out in np.unique(days):
        test = recorded & (days == held_out)
        train = recorded & (days != held_out)
        if test.sum() < MIN_FRAMES or train.sum() < MIN_FRAMES:
            continue

        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(features[train], target[train])
        predicted = model.predict(features[test])

        model_errors.append(np.abs(predicted - target[test]).mean())
        baseline_errors.append(np.abs(target[train].mean() - target[test]).mean())
        correlations.append(correlation(target[test], predicted))

    return model_errors, baseline_errors, correlations


def main():
    data = load()
    features = summary_features(data["spectrograms"])
    probs = data["label_probs"].astype(np.float32)
    days, names = data["day"], data["target_names"]

    print(f"blocks: {len(features):,}   days: {len(np.unique(days))}\n")
    print(f"{'class':<14}{'recorded':>10}{'mean prob':>11}"
          f"{'baseline MAE':>14}{'model MAE':>11}{'r':>7}{'days':>6}")

    rows = []
    for column, name in enumerate(names):
        target = probs[:, column]
        recorded = target > 0.0

        model_errors, baseline_errors, correlations = score_class(
            features, target, days, recorded)

        if not model_errors:
            print(f"{name:<14}{recorded.mean():>9.1%}{'  too rare to score':>38}")
            continue

        mae, baseline = np.mean(model_errors), np.mean(baseline_errors)
        r = np.nanmean(correlations)
        print(f"{name:<14}{recorded.mean():>9.1%}{target[recorded].mean():>11.3f}"
              f"{baseline:>14.3f}{mae:>11.3f}{r:>7.2f}{len(model_errors):>6}")
        rows.append([name, round(float(recorded.mean()), 4),
                     round(float(target[recorded].mean()), 4), round(float(baseline), 4),
                     round(float(mae), 4), round(float(r), 3), len(model_errors)])

    write_report("predict_probability",
                 ["class", "recorded_fraction", "mean_prob_when_recorded",
                  "baseline_mae", "model_mae", "correlation", "days"], rows)


if __name__ == "__main__":
    main()
