"""A small convolutional net over the log-mel spectrogram, for comparison.

Same split, same metrics, same threshold rule as train.py, so the numbers sit
next to each other honestly.

Two things a convnet can do that per-class random forests cannot: see the shape
of a sound across time and frequency rather than a mean and a spread, and learn
one shared representation for every class at once, which is the only real hope
for classes with a few hundred examples.

Whether that is enough is what this file measures. Door, Dog and Cat have
1,112, 640 and 286 positive blocks in the whole corpus.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score
from torch import nn

from dataset import CLASSES, load_spectrograms
from table import table
from train import MIN_EXAMPLES, best_threshold, split

EPOCHS = 30
BATCH = 256
LEARNING_RATE = 3e-4
SEED = 0

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SpectrogramNet(nn.Module):
    """Three convolutional blocks, then one score per class."""

    def __init__(self, n_classes):
        super().__init__()

        def block(channels_in, channels_out):
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, padding=1),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(),
                nn.MaxPool2d(2))

        self.body = nn.Sequential(block(1, 16), block(16, 32), block(32, 64),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, spectrograms):
        return self.head(self.body(spectrograms.unsqueeze(1)))


def normalise(X, reference):
    """Standardise against the training split only, so test stays unseen."""
    mean, std = reference.mean(), reference.std()
    return torch.from_numpy(((X - mean) / std).astype(np.float32))


def train_network(X_train, y_train, X_valid, y_valid, n_classes):
    """Fit, keeping whichever epoch ranks best on validation."""
    torch.manual_seed(SEED)
    model = SpectrogramNet(n_classes).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Rare classes need their positives to count for more, or the net predicts "no" forever.
    positives = y_train.sum(axis=0)
    weight = torch.tensor(((len(y_train) - positives) / np.maximum(positives, 1)),
                          dtype=torch.float32, device=DEVICE)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weight)

    inputs = X_train.to(DEVICE)
    targets = torch.from_numpy(y_train.astype(np.float32)).to(DEVICE)
    best = (-1.0, {key: value.clone() for key, value in model.state_dict().items()})

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(inputs), device=DEVICE)
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            optimiser.zero_grad()
            loss_function(model(inputs[batch]), targets[batch]).backward()
            optimiser.step()

        scores = predict(model, X_valid)
        ranked = np.mean([average_precision_score(y_valid[:, c], scores[:, c])
                          for c in range(n_classes) if y_valid[:, c].sum() >= MIN_EXAMPLES])
        if ranked > best[0]:
            best = (ranked, {k: v.clone() for k, v in model.state_dict().items()})
        print(f"  epoch {epoch + 1:2d}/{EPOCHS}  validation AP {ranked:.3f}"
              f"{'  <- best so far' if ranked == best[0] else ''}")

    model.load_state_dict(best[1])
    return model, best[0]


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = [torch.sigmoid(model(X[start : start + 1024].to(DEVICE))).cpu().numpy()
           for start in range(0, len(X), 1024)]
    return np.concatenate(out)


def main():
    spectrograms, y, days = load_spectrograms()
    train, valid, test = split(days)
    print(f"{len(spectrograms):,} rows of {spectrograms.shape[1]}x{spectrograms.shape[2]} "
          f"log-mel, training on {DEVICE}\n")

    reference = spectrograms[train].astype(np.float32)
    X = {name: normalise(spectrograms[rows].astype(np.float32), reference)
         for name, rows in [("train", train), ("valid", valid), ("test", test)]}

    model, valid_ap = train_network(X["train"], y[train], X["valid"], y[valid], len(CLASSES))
    valid_scores, test_scores = predict(model, X["valid"]), predict(model, X["test"])

    rows = []
    for column, name in enumerate(CLASSES):
        counts = [y[rows_, column].sum() for rows_ in (train, valid, test)]
        if min(counts) < MIN_EXAMPLES:
            continue
        threshold = best_threshold(y[valid, column], valid_scores[:, column])
        predicted = test_scores[:, column] >= threshold
        precision = precision_score(y[test, column], predicted, zero_division=0)
        recall = recall_score(y[test, column], predicted, zero_division=0)
        rows.append([name,
                     f"{y[test, column].mean():.3f}",
                     f"{average_precision_score(y[test, column], test_scores[:, column]):.3f}",
                     f"{precision:.3f}", f"{recall:.3f}",
                     f"{2 * precision * recall / max(precision + recall, 1e-9):.3f}"])

    print(f"\nbest validation AP {valid_ap:.3f}\n")
    table(["class", "chance", "test AP", "prec", "recall", "F1"], rows)
    print("\ncompare with train.py, which uses the same split and the same threshold rule.")


if __name__ == "__main__":
    main()
