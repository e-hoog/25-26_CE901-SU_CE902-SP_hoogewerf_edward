import os
import glob
import csv
import json
import random

import numpy as np
import torch

from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# Settings

# Change for dataset folder
DATASET_FOLDER = os.path.join("data", "graphs")
RESULTS_FOLDER = os.path.join("results")

SEED = 42

HIDDEN_CHANNELS = 32
HEADS = 4
INPUT_FEATURES = 14

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

EPOCHS = 50
DROPOUT = 0.3
BATCH_SIZE = 16

HELDOUT_FRACTION = 0.20

# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load every trial graph for one split ("train" or "test") across both classes

def _subject_seed(subject):
    return sum(ord(char) for char in subject)


def _select_subject_files_for_split(subject, files, split):
    if len(files) < 2:
        return []

    indices = list(range(len(files)))
    rng = random.Random(SEED + _subject_seed(subject))
    rng.shuffle(indices)

    heldout_count = max(1, int(round(len(files) * HELDOUT_FRACTION)))
    heldout_count = min(heldout_count, len(files) - 1)

    test_indices = set(indices[:heldout_count])

    if split == "test":
        return [files[index] for index in range(len(files)) if index in test_indices]
    if split == "train":
        return [files[index] for index in range(len(files)) if index not in test_indices]

    raise ValueError(f"Unknown split: {split}")


def _load_graph_as_data(filename, split_name):
    graph = torch.load(filename, weights_only=False)

    data = Data(
        x=graph["x"],
        edge_index=graph["edge_index"],
        edge_attr=graph["edge_weight"].unsqueeze(1),
        y=graph["y"],
    )
    data.subject = graph["subject"]
    data.recording = graph["recording"]
    data.source_file = graph["source_file"]
    data.trial_key = graph["trial_key"]
    data.split = graph.get("split", split_name)

    return data


def load_split(split):
    files = []

    # Layout A: DATASET_FOLDER/train|test/Alcoholic|Control/*.pt
    for class_name in ["Alcoholic", "Control"]:
        files.extend(glob.glob(os.path.join(DATASET_FOLDER, split, class_name, "*.pt")))

    # Layout B fallback: DATASET_FOLDER/<subject>/*.pt with deterministic split
    if not files:
        for subject_folder in sorted(glob.glob(os.path.join(DATASET_FOLDER, "*"))):
            if not os.path.isdir(subject_folder):
                continue

            subject = os.path.basename(subject_folder)
            if subject == "train" or subject == "test":
                continue

            subject_files = sorted(glob.glob(os.path.join(subject_folder, "*.pt")))
            files.extend(_select_subject_files_for_split(subject, subject_files, split))

    files.sort()

    graphs = []
    for filename in files:
        graphs.append(_load_graph_as_data(filename, split))

    return graphs


# Confirm no trial appears in both splits (subject overlap is expected/allowed here)

def validate_split(train_graphs, test_graphs):
    train_keys = {g.trial_key for g in train_graphs}
    test_keys = {g.trial_key for g in test_graphs}
    overlap = train_keys & test_keys

    if overlap:
        raise RuntimeError(f"TRIAL-LEVEL DATA LEAKAGE: {sorted(overlap)[:20]}")

    train_subjects = {g.subject for g in train_graphs}
    test_subjects = {g.subject for g in test_graphs}

    print(f"Training subjects: {len(train_subjects)}")
    print(f"Test subjects:     {len(test_subjects)}")
    print(f"Subjects in both:  {len(train_subjects & test_subjects)}")
    print(f"Training trials:   {len(train_graphs)}")
    print(f"Held-out trials:   {len(test_graphs)}")
    print()

    if overlap:
        raise RuntimeError("Trial overlap remains.")

    print("PASS: no identical subject+trial appears in both sets.")
    print("PASS: subject overlap is permitted for held-out-trial evaluation.")


class EEGGAT(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels, heads, dropout):
        super().__init__()

        self.dropout = dropout

        self.gat1 = GATConv(
            in_channels=input_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
        )

        self.gat2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            dropout=dropout,
            edge_dim=1,
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 2),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index, edge_attr)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x)


# Train the GAT on all training trials (held-out-trial split, not LOSO)

def train_model(train_graphs):
    loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)

    model = EEGGAT(INPUT_FEATURES, HIDDEN_CHANNELS, HEADS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    print()
    print("======================================================")
    print("TRAINING")
    print("======================================================")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        average_loss = total_loss / len(train_graphs)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Loss: {average_loss:.4f}")

    return model


# Run the trained model on the held-out test trials

def predict(model, test_graphs):
    loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = output.argmax(dim=1)

            predictions.extend(pred.cpu().numpy().tolist())
            labels.extend(batch.y.cpu().numpy().tolist())
            # Metadata is kept in the original test graph order, so these
            # predictions still line up with test_graphs afterwards.

    return np.array(predictions), np.array(labels)


# Combine trial-level predictions into a majority vote per subject

def subject_results(test_graphs, predictions, labels):
    grouped = defaultdict(list)
    for i, graph in enumerate(test_graphs):
        grouped[graph.subject].append(i)

    results = []
    for subject in sorted(grouped):
        indices = grouped[subject]
        subject_preds = predictions[indices]
        subject_labels = labels[indices]

        majority_prediction = int(np.mean(subject_preds) >= 0.5)
        true_label = int(subject_labels[0])

        results.append({
            "subject": subject,
            "true": true_label,
            "prediction": majority_prediction,
            "correct": majority_prediction == true_label,
            "n_trials": len(indices),
            "trial_accuracy": float(np.mean(subject_preds == subject_labels)),
        })

    return results


# Save one row per trial: prediction vs true label

def save_trial_results(test_graphs, predictions, labels):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    path = os.path.join(RESULTS_FOLDER, "heldout_trial_predictions.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["TrialKey", "Subject", "Recording", "TrueLabel", "Prediction", "Correct"])

        for graph, pred, label in zip(test_graphs, predictions, labels):
            writer.writerow(
                [graph.trial_key, graph.subject, graph.recording, int(label), int(pred), int(pred == label)]
            )

    return path


# Save one row per subject: majority-vote prediction and trial accuracy

def save_subject_results(results):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    path = os.path.join(RESULTS_FOLDER, "heldout_trial_subject_summary.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "TrueLabel", "Prediction", "Correct", "HeldOutTrials", "TrialAccuracy"])

        for r in results:
            writer.writerow(
                [r["subject"], r["true"], r["prediction"], r["correct"], r["n_trials"], r["trial_accuracy"]]
            )

    return path


def save_json_results(
    predictions,
    labels,
    accuracy,
    balanced,
    sensitivity,
    specificity,
    tn,
    fp,
    fn,
    tp,
    subject_summary,
    subject_accuracy,
    subject_balanced,
    subject_sensitivity,
    subject_specificity,
    stn,
    sfp,
    sfn,
    stp,
):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    output = {
        "configuration": {
            "dataset": DATASET_FOLDER,
            "model": "GAT held-out trial",
            "input_features": INPUT_FEATURES,
            "hidden_channels": HIDDEN_CHANNELS,
            "heads": HEADS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "seed": SEED,
            "device": str(DEVICE),
        },
        "metrics": {
            "trial_level": {
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "n_trials": int(len(labels)),
            },
            "subject_level": {
                "accuracy": float(subject_accuracy),
                "balanced_accuracy": float(subject_balanced),
                "sensitivity": float(subject_sensitivity),
                "specificity": float(subject_specificity),
                "true_negative": int(stn),
                "false_positive": int(sfp),
                "false_negative": int(sfn),
                "true_positive": int(stp),
                "n_subjects": int(len(subject_summary)),
            },
        },
        "results": {
            "trial_predictions": [
                {
                    "index": int(i),
                    "actual": int(label),
                    "prediction": int(prediction),
                    "correct": bool(prediction == label),
                }
                for i, (prediction, label) in enumerate(zip(predictions, labels), start=1)
            ],
            "subject_summary": subject_summary,
        },
    }

    path = os.path.join(RESULTS_FOLDER, "heldout_trial_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return path


def main():
    print("======================================================")
    print("EEG GAT — HELD-OUT TRIAL EXPERIMENT")
    print("======================================================")
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print()

    train_graphs = load_split("train")
    test_graphs = load_split("test")

    if not train_graphs:
        raise RuntimeError(
            "No training graphs found. Expected either:\n"
            "1) data/graphs/train/Alcoholic|Control/*.pt\n"
            "2) data/graphs/<subject>/*.pt with deterministic held-out split"
        )
    if not test_graphs:
        raise RuntimeError(
            "No test graphs found. Expected either:\n"
            "1) data/graphs/test/Alcoholic|Control/*.pt\n"
            "2) data/graphs/<subject>/*.pt with deterministic held-out split"
        )

    validate_split(train_graphs, test_graphs)
    print()

    train_labels = np.array([int(g.y) for g in train_graphs])
    test_labels = np.array([int(g.y) for g in test_graphs])

    print(f"Training alcoholic trials: {np.sum(train_labels == 1)}")
    print(f"Training control trials:   {np.sum(train_labels == 0)}")
    print(f"Test alcoholic trials:     {np.sum(test_labels == 1)}")
    print(f"Test control trials:       {np.sum(test_labels == 0)}")

    model = train_model(train_graphs)

    predictions, labels = predict(model, test_graphs)

    # Trial-level metrics
    accuracy = accuracy_score(labels, predictions)
    balanced = balanced_accuracy_score(labels, predictions)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Subject-level aggregation: majority vote across a subject's held-out trials
    subject_summary = subject_results(test_graphs, predictions, labels)

    subject_predictions = np.array([r["prediction"] for r in subject_summary])
    subject_labels = np.array([r["true"] for r in subject_summary])

    subject_accuracy = accuracy_score(subject_labels, subject_predictions)
    subject_balanced = balanced_accuracy_score(subject_labels, subject_predictions)

    subject_cm = confusion_matrix(subject_labels, subject_predictions, labels=[0, 1])
    stn, sfp, sfn, stp = subject_cm.ravel()

    subject_sensitivity = stp / (stp + sfn) if (stp + sfn) > 0 else 0.0
    subject_specificity = stn / (stn + sfp) if (stn + sfp) > 0 else 0.0

    print()
    print()
    print("======================================================")
    print("FINAL HELD-OUT TRIAL RESULTS")
    print("======================================================")

    print()
    print("TRIAL LEVEL")
    print("----------------------------------------------")
    print(f"Accuracy:       {accuracy * 100:.2f}%")
    print(f"Balanced Acc.:  {balanced * 100:.2f}%")
    print(f"Sensitivity:    {sensitivity * 100:.2f}%")
    print(f"Specificity:    {specificity * 100:.2f}%")

    print()
    print("SUBJECT AGGREGATION")
    print("----------------------------------------------")
    print(f"Subjects:       {len(subject_summary)}")
    print(f"Accuracy:       {subject_accuracy * 100:.2f}%")
    print(f"Balanced Acc.:  {subject_balanced * 100:.2f}%")
    print(f"Sensitivity:    {subject_sensitivity * 100:.2f}%")
    print(f"Specificity:    {subject_specificity * 100:.2f}%")

    print()
    print("TRIAL CONFUSION MATRIX")
    print("----------------------------------------------")
    print("                 Pred Control   Pred Alcoholic")
    print(f"Actual Control       {tn:5d}          {fp:5d}")
    print(f"Actual Alcoholic     {fn:5d}          {tp:5d}")

    print()
    print("SUBJECT CONFUSION MATRIX")
    print("----------------------------------------------")
    print("                 Pred Control   Pred Alcoholic")
    print(f"Actual Control       {stn:5d}          {sfp:5d}")
    print(f"Actual Alcoholic     {sfn:5d}          {stp:5d}")

    print()
    print("PER-SUBJECT RESULTS")
    print("----------------------------------------------")

    for r in subject_summary:
        actual = "Alcoholic" if r["true"] else "Control"
        predicted = "Alcoholic" if r["prediction"] else "Control"
        status = "CORRECT" if r["correct"] else "WRONG"

        print(
            f"{r['subject']} | "
            f"Actual: {actual:<9} | "
            f"Predicted: {predicted:<9} | "
            f"Trials: {r['n_trials']:3d} | "
            f"Trial Acc: {r['trial_accuracy'] * 100:6.2f}% | "
            f"{status}"
        )

    trial_file = save_trial_results(test_graphs, predictions, labels)
    subject_file = save_subject_results(subject_summary)
    json_file = save_json_results(
        predictions=predictions,
        labels=labels,
        accuracy=accuracy,
        balanced=balanced,
        sensitivity=sensitivity,
        specificity=specificity,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        subject_summary=subject_summary,
        subject_accuracy=subject_accuracy,
        subject_balanced=subject_balanced,
        subject_sensitivity=subject_sensitivity,
        subject_specificity=subject_specificity,
        stn=stn,
        sfp=sfp,
        sfn=sfn,
        stp=stp,
    )

    print()
    print("======================================================")
    print("FILES SAVED")
    print("======================================================")
    print(trial_file)
    print(subject_file)
    print(json_file)

    print()

    print()
    print("======================================================")
    print("HELD-OUT TRIAL GAT TEST COMPLETE")
    print("======================================================")


if __name__ == "__main__":
    main()