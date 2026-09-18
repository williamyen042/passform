"""Train and cross-validate the pass-quality classifier. Phases 2-4 of the plan.

    .venv/bin/python scripts/train_classifier.py

Reports three numbers side by side, which is the whole point: the rule-based
scorer already in core/scorer.py, logistic regression, and a random forest.
A learned model that cannot beat the rules it replaces is a finding, not a
failure, and it should be as easy to see as the flattering case.

Deliberately not a deep model: at this sample size a net would overfit and
explain nothing. See classifier_plan.md 3.1.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Bookkeeping, not evidence: these describe the rep rather than the movement,
# and training on them would let the model read the labeller instead of the
# pass. Everything else in features.csv is a measurement.
NOT_FEATURES = {"rep_id", "quality", "position", "source_video", "filename",
                "duration", "candidates", "contact_source", "contact_frac",
                "baseline_hint", "score_overall", "score_stability",
                "score_integrity", "score_kinetic"}
FOLDS = 5
SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path,
                        default=Path("volleyball_dataset/features.csv"))
    parser.add_argument("--binary", action="store_true",
                        help="Collapse to in-system (3,2) vs out (1,0). The "
                             "fallback in the plan if the four classes are "
                             "too thin to fit.")
    return parser.parse_args()


def models():
    """Scaling matters for the regression and not for the forest, so each
    model carries its own preprocessing rather than sharing a global one."""
    impute = SimpleImputer(strategy="median")
    return {
        "logistic regression": make_pipeline(
            impute, StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random forest": make_pipeline(
            impute,
            RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                   class_weight="balanced", random_state=SEED)),
        "majority class": make_pipeline(impute, DummyClassifier(strategy="prior")),
    }


def report(name, truth, predicted, labels):
    accuracy = accuracy_score(truth, predicted)
    # Mean absolute error, because the label is ordinal: calling a 3 a 2 is a
    # near miss and calling a 0 a 3 is not, and accuracy scores them the same.
    mae = np.mean(np.abs(np.asarray(truth) - np.asarray(predicted)))
    print(f"\n{name}")
    print(f"  accuracy {accuracy:.3f}   mean absolute error {mae:.2f}")
    print("  " + classification_report(
        truth, predicted, labels=labels, zero_division=0,
        target_names=[f"q{label}" for label in labels]).replace("\n", "\n  "))
    print("  confusion (rows = true, cols = predicted, order "
          f"{[int(label) for label in labels]}):")
    for label, row in zip(labels, confusion_matrix(truth, predicted, labels=labels)):
        print(f"    q{label}  {row}")
    return accuracy, mae


def per_fold(model, features, target, splitter, groups=None):
    """Accuracy per fold, so the spread can be reported alongside the mean.

    One number from a single split on this many reps is mostly noise; the
    standard deviation across folds is the honest part of the result.
    """
    scores = []
    for train, test in splitter.split(features, target, groups):
        fitted = model.fit(features.iloc[train], target.iloc[train])
        scores.append(accuracy_score(target.iloc[test],
                                     fitted.predict(features.iloc[test])))
    return np.array(scores)


def main():
    args = parse_args()
    frame = pd.read_csv(args.features)
    measured = frame[frame["candidates"] > 0]
    dropped = len(frame) - len(measured)

    target = measured["quality"]
    if args.binary:
        target = (target >= 2).astype(int)
    features = measured.drop(columns=[c for c in measured.columns if c in NOT_FEATURES])
    labels = sorted(target.unique())

    print(f"{len(frame)} labelled reps, {dropped} with no detected contact, "
          f"{len(measured)} usable")
    print(f"{features.shape[1]} features: {', '.join(features.columns)}")
    print("class counts:", target.value_counts().sort_index().to_dict())
    print("source videos:", measured["source_video"].value_counts().to_dict())

    # Phase 2: the rule-based scorer on the same reps, before anything is
    # trained. Every learned number below is only interesting relative to this.
    baseline = measured["baseline_hint"]
    if args.binary:
        baseline = (baseline >= 2).astype(int)
    report("BASELINE - rule-based scorer (core/scorer.py)", target, baseline, labels)

    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for name, model in models().items():
        predicted = cross_val_predict(model, features, target, cv=splitter)
        scores = per_fold(model, features, target, splitter)
        report(f"{name} ({FOLDS}-fold stratified CV)", target, predicted, labels)
        print(f"  per-fold accuracy {np.round(scores, 3)} -> "
              f"{scores.mean():.3f} +/- {scores.std():.3f}")

    # Harder question than the CV above: does it hold up on a session it has
    # never seen? Reps from one clip share a camera, a gym and often a passer,
    # so a random split lets the model recognise the session instead of the
    # pass. This is the number that predicts what happens on new footage.
    groups = measured["source_video"]
    if groups.nunique() > 1:
        print(f"\n--- leave-one-video-out ({groups.nunique()} videos) ---")
        for name, model in models().items():
            scores = per_fold(model, features, target, LeaveOneGroupOut(), groups)
            print(f"  {name:<22} {np.round(scores, 3)} -> "
                  f"{scores.mean():.3f} +/- {scores.std():.3f}")

    # Phase 4: which measurements actually carry the signal.
    forest = models()["random forest"].fit(features, target)
    importances = pd.Series(
        forest[-1].feature_importances_, index=features.columns).nlargest(10)
    print("\ntop features (random forest, fitted on everything - "
          "for reading, not for scoring):")
    for feature, weight in importances.items():
        print(f"  {weight:.3f}  {feature}")


if __name__ == "__main__":
    main()
