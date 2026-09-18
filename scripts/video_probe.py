"""Fit a classifier on the VideoMAE embeddings, and compare it to the features.

    PYTHONPATH=. .venv/bin/python scripts/video_probe.py

Same protocol as scripts/train_classifier.py - 5-fold stratified CV, the same
majority-class baseline, leave-one-session-out - so the number here can be put
beside the ball-feature number honestly rather than being a different
experiment that happens to produce a percentage.

768 dimensions against 86 reps is nine dimensions per sample, so the probe is
strongly regularised and PCA is offered: without one of the two it fits the
clips rather than the passing.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("volleyball_dataset"))
    parser.add_argument("--components", type=int, default=24,
                        help="PCA dimensions before the classifier. 0 to skip.")
    return parser.parse_args()


def main():
    args = parse_args()
    cached = np.load(args.dataset / "video_embeddings.npz")
    embeddings = {int(key): cached[key] for key in cached.files}
    rows = [row for row in csv.DictReader((args.dataset / "labels.csv").open())
            if int(row["rep_id"]) in embeddings]

    X = np.stack([embeddings[int(row["rep_id"])] for row in rows])
    y = np.array([int(row["quality"]) for row in rows])
    videos = np.array([row["source_video"] for row in rows])
    print(f"{len(rows)} clips, {X.shape[1]} dimensions each")
    print(f"class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"sources: {dict(zip(*np.unique(videos, return_counts=True)))}\n")

    head = ([PCA(n_components=args.components, random_state=0)]
            if args.components else [])
    models = {
        "logistic (regularised)": make_pipeline(
            StandardScaler(), *head,
            LogisticRegression(max_iter=5000, C=0.05, class_weight="balanced")),
        "linear SVC": make_pipeline(
            StandardScaler(), *head,
            LinearSVC(C=0.01, class_weight="balanced")),
        "majority class": make_pipeline(DummyClassifier(strategy="prior")),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    print(f"{'model':<24} {'accuracy':>9} {'spread':>8} {'within 1':>9} {'MAE':>6}")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv)
        predicted = cross_val_predict(model, X, y, cv=cv)
        within = (np.abs(predicted - y) <= 1).mean()
        mae = np.abs(predicted - y).mean()
        print(f"{name:<24} {scores.mean():>9.3f} {scores.std():>8.3f} "
              f"{within:>9.3f} {mae:>6.2f}")

    if len(set(videos)) > 1:
        print(f"\nleave-one-session-out ({len(set(videos))} videos):")
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=LeaveOneGroupOut(), groups=videos)
            print(f"  {name:<24} {np.round(scores, 3)} -> {scores.mean():.3f}")

    # The embeddings know which session a clip came from whether or not they
    # know anything about passing. If they separate sessions far better than
    # labels, the probe above is partly reading the gym.
    session = cross_val_score(models["logistic (regularised)"], X, videos, cv=cv).mean()
    common = np.unique(videos, return_counts=True)[1].max() / len(videos)
    print(f"\n  sanity check - predicting the source video from the same "
          f"embeddings: {session:.3f} (majority {common:.3f})")


if __name__ == "__main__":
    main()
