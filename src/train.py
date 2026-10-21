"""Pick the best model for each class, then measure it once on held-out audio.

A fixed three-way split by day, chronological so the choice is not cherry-picked:

    train       16-20 Nov    fits every candidate
    validation  21 Nov       picks which candidate, and where to put its cutoff
    test        22 Nov       read once, at the end

Splitting by day rather than at random matters: rows are 0.213 s apart, so
blocks from one passing car are near duplicates, and a random split would test
the model on audio it had already seen.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import CLASSES, load
from table import table

MODEL = Path("data/tagger.joblib")

TRAIN_DAYS = ["20231116", "20231117", "20231118", "20231119", "20231120"]
VALID_DAY = "20231121"
TEST_DAY = "20231122"

MIN_EXAMPLES = 30  # a class needs this many positives in every split to be scored

CANDIDATES = {
    "logistic": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "forest": lambda: RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0),
    "boosting": lambda: HistGradientBoostingClassifier(random_state=0),
    "boosting deep": lambda: HistGradientBoostingClassifier(
        random_state=0, max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=False),
}


def split(days):
    return np.isin(days, TRAIN_DAYS), days == VALID_DAY, days == TEST_DAY


def best_threshold(truth, scores):
    """The cutoff with the best F1 on validation. Never chosen on test."""
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return thresholds[min(int(np.argmax(f1)), len(thresholds) - 1)]


def select_on_validation(X, target, train, valid):
    """Fit every candidate on train, keep whichever ranks best on validation."""
    def score(name):
        model = CANDIDATES[name]().fit(X[train], target[train])
        scores = model.predict_proba(X[valid])[:, 1]
        return average_precision_score(target[valid], scores), name, model, scores

    return max((score(name) for name in CANDIDATES), key=lambda result: result[0])


def evaluate(X, target, train, valid, test):
    """Choose on validation, then read test once."""
    valid_ap, chosen, model, valid_scores = select_on_validation(X, target, train, valid)
    threshold = best_threshold(target[valid], valid_scores)

    scores = model.predict_proba(X[test])[:, 1]
    precision = precision_score(target[test], scores >= threshold, zero_division=0)
    recall = recall_score(target[test], scores >= threshold, zero_division=0)

    return {
        "model": chosen,
        "threshold": threshold,
        "valid AP": valid_ap,
        "chance": target[test].mean(),
        "test AP": average_precision_score(target[test], scores),
        "precision": precision,
        "recall": recall,
        "F1": 2 * precision * recall / max(precision + recall, 1e-9),
    }


def main():
    X, y, days = load()
    train, valid, test = split(days)

    print(f"{len(X):,} rows, {X.shape[1]} features\n")
    table(["split", "rows", "days"],
          [[label, f"{rows.sum():,}", " ".join(sorted(set(days[rows])))]
           for label, rows in [("train", train), ("validation", valid), ("test", test)]])

    results, dropped = {}, []
    for column, name in enumerate(CLASSES):
        counts = [int(y[rows, column].sum()) for rows in (train, valid, test)]
        if min(counts) < MIN_EXAMPLES:
            dropped.append(f"{name} ({'/'.join(map(str, counts))})")
            continue
        results[name] = evaluate(X, y[:, column], train, valid, test)

    print()
    table(["class", "model", "valid AP", "chance", "test AP", "prec", "recall", "F1"],
          [[name, r["model"]] + [f"{r[key]:.3f}" for key in
                                 ("valid AP", "chance", "test AP", "precision", "recall", "F1")]
           for name, r in results.items()])

    if dropped:
        print(f"\ndropped, too few positives in train/valid/test: {', '.join(dropped)}")
    print("chance is what a coin weighted to the class frequency scores on test.")

    # Ship each chosen configuration, refitted on train + validation. Test stays
    # unread, so the scores above still describe what is shipped.
    fitted = train | valid
    joblib.dump(
        {"classes": list(results),
         "thresholds": [r["threshold"] for r in results.values()],
         "models": [CANDIDATES[r["model"]]().fit(X[fitted], y[fitted, CLASSES.index(name)])
                    for name, r in results.items()]},
        MODEL)
    print(f"\nwrote {MODEL} with {len(results)} of {len(CLASSES)} classes")


if __name__ == "__main__":
    main()
