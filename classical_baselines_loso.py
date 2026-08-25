import os
import json
import random

import numpy as np

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# Settings

# Change for dataset folder
DATASET_FOLDER = "data\\graphs"
RESULTS_FOLDER = "results"

SEED = 42

# Classifiers

C = 1.0
MAX_ITER = 5000

CLASSIFIERS = ("logistic_regression", "linear_svm", "rbf_svm")

CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF SVM",
}

# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Build a fresh, untrained classifier of the given kind

def build_classifier(name):
    if name == "logistic_regression":
        return LogisticRegression(C=C, max_iter=MAX_ITER, solver="liblinear", random_state=SEED)
    if name == "linear_svm":
        return SVC(kernel="linear", C=C, probability=True, random_state=SEED)
    if name == "rbf_svm":
        return SVC(kernel="rbf", C=C, probability=True, random_state=SEED)
    raise ValueError(f"Unknown classifier: {name}")


# Find every subject's saved trial graphs

def discover_subjects():
    subjects = {}

    # Layout A: DATASET_FOLDER/train/Alcoholic|Control/*.pt
    for class_name in ["Alcoholic", "Control"]:
        folder = os.path.join(DATASET_FOLDER, "train", class_name)
        if not os.path.exists(folder):
            continue

        for filename in os.listdir(folder):
            if not filename.endswith(".pt"):
                continue

            subject = filename.split("_rd")[0]
            path = os.path.join(folder, filename)
            graph = torch.load(path, map_location="cpu", weights_only=False)
            label = int(graph["y"].item())

            subjects.setdefault(subject, {"label": label, "files": []})
            subjects[subject]["files"].append(path)

    # Layout B: DATASET_FOLDER/<subject>/*.pt with optional metadata.json
    if not subjects and os.path.exists(DATASET_FOLDER):
        for entry in sorted(os.listdir(DATASET_FOLDER)):
            if entry == "metadata.json":
                continue

            subject_folder = os.path.join(DATASET_FOLDER, entry)
            if not os.path.isdir(subject_folder):
                continue

            pt_files = sorted(
                os.path.join(subject_folder, filename)
                for filename in os.listdir(subject_folder)
                if filename.endswith(".pt")
            )

            if not pt_files:
                continue

            first_graph = torch.load(pt_files[0], map_location="cpu", weights_only=False)
            label = int(first_graph["y"].item())

            subjects[entry] = {
                "label": label,
                "files": pt_files,
            }

    for subject in subjects:
        subjects[subject]["files"] = sorted(subjects[subject]["files"])

    return subjects


# Load and flatten a subject's per-trial node features

def load_subject_features(subject, subjects):
    trial_features = []

    for path in subjects[subject]["files"]:
        graph = torch.load(path, map_location="cpu", weights_only=False)

        x = graph["x"].numpy()
        if x.shape != (61, 14):
            raise ValueError(f"{subject}: unexpected feature shape {x.shape}")

        flattened = x.reshape(-1)
        trial_features.append(flattened)

    trial_features = np.stack(trial_features, axis=0)
    return trial_features


def aggregate_subject(trial_features):
    return trial_features.mean(axis=0)


# Compute accuracy, balanced accuracy, sensitivity and specificity for one classifier's results

def compute_metrics(results):
    y_true = np.array([result["actual"] for result in results])
    y_pred = np.array([result["prediction"] for result in results])

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    print()
    print("======================================================")
    print("FEATURE-ONLY SUBJECT-LEVEL LOSO")
    print("======================================================")
    print(f"Dataset: {DATASET_FOLDER}")
    print("Representation: 61 × 14 engineered features")
    print("Connectivity: NONE")
    print("Aggregation: mean across trials")
    print("Classifiers: Logistic Regression, Linear SVM, RBF SVM")
    print()

    subjects = discover_subjects()
    subject_list = sorted(subjects.keys())
    print(f"Subjects: {len(subject_list)}")
    print()

    if not subject_list:
        raise RuntimeError(
            "No subjects found. Expected one of these layouts:\n"
            "1) data/graphs/train/Alcoholic|Control/*.pt\n"
            "2) data/graphs/<subject>/*.pt"
        )

    if len(subject_list) < 2:
        raise RuntimeError(
            "Need at least 2 subjects for LOSO evaluation. "
            f"Found only {len(subject_list)}."
        )

    # Aggregate every subject's trials into one feature vector, shared across classifiers
    subject_features = {}
    subject_labels = {}
    for subject in subject_list:
        trial_features = load_subject_features(subject, subjects)
        subject_features[subject] = aggregate_subject(trial_features)
        subject_labels[subject] = subjects[subject]["label"]

    # Leave-one-subject-out cross-validation
    results = {name: [] for name in CLASSIFIERS}

    for fold, test_subject in enumerate(subject_list, start=1):
        print()
        print("======================================================")
        print(f"FOLD {fold}/{len(subject_list)}")
        print(f"Test subject: {test_subject}")
        print("======================================================")

        train_subjects = [subject for subject in subject_list if subject != test_subject]

        X_train = np.stack([subject_features[subject] for subject in train_subjects])
        y_train = np.array([subject_labels[subject] for subject in train_subjects])

        X_test = subject_features[test_subject][None, :]
        y_test = np.array([subject_labels[test_subject]])

        # Fit the scaler on training subjects only, shared by every classifier this fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        actual = int(y_test[0])
        actual_name = "Alcoholic" if actual == 1 else "Control"

        print()
        print(f"Subject: {test_subject}")
        print(f"Actual:  {actual_name}")
        print(f"Trials:  {len(subjects[test_subject]['files'])}")
        print()

        for name in CLASSIFIERS:
            model = build_classifier(name)
            model.fit(X_train_scaled, y_train)

            prediction = int(model.predict(X_test_scaled)[0])
            probabilities = model.predict_proba(X_test_scaled)[0]
            probability_control = float(probabilities[0])
            probability_alcoholic = float(probabilities[1])

            correct = (prediction == actual)
            predicted_name = "Alcoholic" if prediction == 1 else "Control"

            print(
                f"  {CLASSIFIER_LABELS[name]:<20s} | "
                f"Predicted: {predicted_name:<9s} | "
                f"Correct: {'YES' if correct else 'NO':<3s} | "
                f"P(Control)={probability_control:.4f} "
                f"P(Alcoholic)={probability_alcoholic:.4f}"
            )

            results[name].append({
                "fold": fold,
                "subject": test_subject,
                "actual": actual,
                "prediction": prediction,
                "probability_control": probability_control,
                "probability_alcoholic": probability_alcoholic,
                "n_trials": len(subjects[test_subject]["files"]),
                "correct": correct,
            })

    # Overall metrics per classifier
    metrics = {name: compute_metrics(results[name]) for name in CLASSIFIERS}

    print()
    print("======================================================")
    print("FEATURE-ONLY SUBJECT-LEVEL LOSO RESULTS")
    print("======================================================")
    print(f"Subjects: {len(subject_list)}")
    print()
    print(f"{'Classifier':<22s} {'Accuracy':>10s} {'Balanced':>10s} {'Sensitivity':>12s} {'Specificity':>12s}")
    for name in CLASSIFIERS:
        m = metrics[name]
        print(
            f"{CLASSIFIER_LABELS[name]:<22s} "
            f"{m['accuracy'] * 100:>9.2f}% "
            f"{m['balanced_accuracy'] * 100:>9.2f}% "
            f"{m['sensitivity'] * 100:>11.2f}% "
            f"{m['specificity'] * 100:>11.2f}%"
        )

    best_name = max(CLASSIFIERS, key=lambda name: metrics[name]["accuracy"])
    print()
    print(f"Best accuracy: {CLASSIFIER_LABELS[best_name]} ({metrics[best_name]['accuracy'] * 100:.2f}%)")

    for name in CLASSIFIERS:
        m = metrics[name]
        print()
        print(f"{CLASSIFIER_LABELS[name]} confusion matrix")
        print("                  Pred Control   Pred Alcoholic")
        print(f"Actual Control       {m['true_negative']:5d}            {m['false_positive']:5d}")
        print(f"Actual Alcoholic     {m['false_negative']:5d}            {m['true_positive']:5d}")

    # Per-subject breakdown, per classifier
    print()
    print("Subject results:")
    for name in CLASSIFIERS:
        print()
        print(f"  {CLASSIFIER_LABELS[name]}:")
        for result in results[name]:
            actual_name = "Alcoholic" if result["actual"] == 1 else "Control"
            predicted_name = "Alcoholic" if result["prediction"] == 1 else "Control"
            marker = "✓" if result["correct"] else "✗"
            print(
                f"    {marker} "
                f"{result['subject']} | "
                f"Actual={actual_name:9s} | "
                f"Predicted={predicted_name:9s} | "
                f"Trials={result['n_trials']:3d}"
            )

    # Save configuration, metrics and per-subject results for every classifier
    output = {
        "configuration": {
            "dataset": DATASET_FOLDER,
            "representation": "61 electrodes × 14 engineered features",
            "n_features": 854,
            "connectivity": None,
            "trial_aggregation": "mean",
            "evaluation": "subject-level LOSO",
            "C": C,
            "max_iter": MAX_ITER,
            "seed": SEED,
        },
        "classifiers": {
            name: {
                "label": CLASSIFIER_LABELS[name],
                "metrics": metrics[name],
                "results": results[name],
            }
            for name in CLASSIFIERS
        },
    }

    output_path = os.path.join(RESULTS_FOLDER, "classical_baseline_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("Results saved to:")
    print(output_path)
    print("======================================================")


if __name__ == "__main__":
    main()