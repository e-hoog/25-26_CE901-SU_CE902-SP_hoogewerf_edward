import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# Settings

# Change for dataset folder
DATASET_FOLDER = "data\\graphs"
RESULTS_FOLDER = "results"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GAT

HIDDEN_CHANNELS = 32
HEADS = 4
DROPOUT = 0.30

# Subject attention

ATTENTION_HIDDEN = 32

# Training

MAX_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 15

# Number of validation subjects, kept identical to the previous LOSO GAT setup
N_VALIDATION_SUBJECTS = 2

# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_metadata():
    metadata_path = os.path.join(DATASET_FOLDER, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find metadata:\n{metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load every saved trial graph belonging to one subject, as PyG Data objects

def load_subject_graphs(subject, metadata):
    if subject not in metadata["subjects"]:
        raise KeyError(f"Subject not found in metadata: {subject}")

    subject_info = metadata["subjects"][subject]
    graphs = []

    for recording in subject_info["trials"]:
        filename = f"{subject}_rd{recording:03d}.pt"
        path = os.path.join(DATASET_FOLDER, subject, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing graph:\n{path}")

        graph = torch.load(path, map_location="cpu", weights_only=False)

        data = Data(
            x=graph["x"].float(),
            edge_index=graph["edge_index"].long(),
            edge_weight=graph["edge_weight"].float(),
            y=graph["y"].long(),
        )
        data.subject = graph["subject"]
        data.recording = graph["recording"]
        data.source_file = graph["source_file"]
        graphs.append(data)

    if not graphs:
        raise RuntimeError(f"Subject {subject} has no trials")

    return graphs


# All subject ids present in the dataset

def get_subjects(metadata):
    subjects = sorted(metadata["subjects"].keys())
    if not subjects:
        raise RuntimeError("No subjects found in metadata")
    return subjects


# Load every subject's graphs into one flat list

def load_graphs(metadata):
    graphs = []
    for subject in get_subjects(metadata):
        graphs.extend(load_subject_graphs(subject, metadata))
    return graphs


# Fit a StandardScaler using only the training subjects' node features

def fit_training_scaler(train_subjects):
    feature_arrays = []
    for subject in train_subjects:
        for graph in subject["graphs"]:
            feature_arrays.append(graph.x.cpu().numpy())

    scaler = StandardScaler()
    scaler.fit(np.concatenate(feature_arrays, axis=0))
    return scaler


# Apply a fitted scaler to a list of graphs

def scale_graphs(graphs, scaler):
    scaled_graphs = []
    for graph in graphs:
        scaled_graph = graph.clone()
        scaled_graph.x = torch.tensor(scaler.transform(graph.x.cpu().numpy()), dtype=torch.float32)
        scaled_graphs.append(scaled_graph)
    return scaled_graphs


# Group a flat list of trial graphs by subject

def group_by_subject(graphs):
    subjects = {}
    for graph in graphs:
        subject = graph.subject
        if subject not in subjects:
            subjects[subject] = {"label": int(graph.y.item()), "graphs": []}
        subjects[subject]["graphs"].append(graph)
    return subjects


class TrialGAT(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.gat1 = GATConv(
            in_channels=num_features,
            out_channels=HIDDEN_CHANNELS,
            heads=HEADS,
            concat=True,
            dropout=DROPOUT,
            edge_dim=1,
        )

        self.gat2 = GATConv(
            in_channels=HIDDEN_CHANNELS * HEADS,
            out_channels=HIDDEN_CHANNELS,
            heads=1,
            concat=False,
            dropout=DROPOUT,
            edge_dim=1,
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index, edge_weight, batch):
        edge_attr = edge_weight.view(-1, 1)

        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # One embedding per trial
        trial_embedding = global_mean_pool(x, batch)
        return trial_embedding


class TrialAttention(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    # trial_embeddings: number_of_trials x embedding_dim
    def forward(self, trial_embeddings):
        scores = self.attention(trial_embeddings).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        subject_embedding = torch.sum(weights.unsqueeze(-1) * trial_embeddings, dim=0)
        return subject_embedding, weights


class GATAttentionModel(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.encoder = TrialGAT(num_features)
        self.attention = TrialAttention(embedding_dim=HIDDEN_CHANNELS, hidden_dim=ATTENTION_HIDDEN)

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_CHANNELS, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 2),
        )

    # Batch a subject's trial graphs together and encode them with the GAT
    def encode_trials(self, graphs):
        loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
        batch = next(iter(loader))
        batch = batch.to(DEVICE)

        trial_embeddings = self.encoder(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        return trial_embeddings

    def forward(self, graphs):
        trial_embeddings = self.encode_trials(graphs)
        subject_embedding, weights = self.attention(trial_embeddings)
        logits = self.classifier(subject_embedding)
        return logits, weights


# Run the trained model on one subject's graphs

def predict_subject(model, graphs, scaler):
    model.eval()
    graphs = scale_graphs(graphs, scaler)

    with torch.no_grad():
        logits, weights = model(graphs)
        probabilities = torch.softmax(logits, dim=0)
        prediction = int(torch.argmax(probabilities).item())

    return prediction, probabilities.cpu().numpy(), weights.cpu().numpy()


# Train the GAT + attention model on the training subjects, using validation
# subjects for early stopping

def train_fold(train_subjects, validation_subjects, test_subject, num_features):
    scaler = fit_training_scaler(train_subjects)

    scaled_train_subjects = [
        {**subject, "graphs": scale_graphs(subject["graphs"], scaler)}
        for subject in train_subjects
    ]

    scaled_validation_subjects = [
        {**subject, "graphs": scale_graphs(subject["graphs"], scaler)}
        for subject in validation_subjects
    ]

    model = GATAttentionModel(num_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Each subject is one training example
        shuffled_subjects = list(scaled_train_subjects)
        random.shuffle(shuffled_subjects)

        for subject in shuffled_subjects:
            graphs = subject["graphs"]
            label = torch.tensor([subject["label"]], dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            logits, _ = model(graphs)
            loss = criterion(logits.unsqueeze(0), label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            prediction = int(torch.argmax(logits).item())
            correct += (prediction == subject["label"])
            total += 1

        train_accuracy = 100.0 * correct / total

        # Validation
        model.eval()
        validation_losses = []
        validation_correct = 0

        with torch.no_grad():
            for subject in scaled_validation_subjects:
                graphs = subject["graphs"]
                label = torch.tensor(subject["label"], dtype=torch.long, device=DEVICE)

                logits, _ = model(graphs)
                loss = criterion(logits, label)
                validation_losses.append(loss.item())

                prediction = int(torch.argmax(logits).item())
                if prediction == subject["label"]:
                    validation_correct += 1

        val_loss = np.mean(validation_losses)
        val_accuracy = 100.0 * validation_correct / len(validation_subjects)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"    Epoch {epoch:3d}/{MAX_EPOCHS} | "
                f"Loss {total_loss / total:.4f} | "
                f"Train {train_accuracy:6.2f}% | "
                f"Val {val_accuracy:6.2f}%"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    print()
    print("======================================================")
    print("GAT + LEARNED TRIAL ATTENTION")
    print("SUBJECT-LEVEL LOSO")
    print("======================================================")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_FOLDER}")
    print("Connectivity: Pearson top-8")
    print("Aggregation: learned trial attention")
    print()

    metadata = load_metadata()
    graphs = load_graphs(metadata)
    num_features = int(metadata["configuration"]["n_features"])

    subjects = group_by_subject(graphs)
    subject_names = sorted(subjects.keys())

    print(f"Subjects: {len(subject_names)}")
    print(f"Node features: {num_features}")

    # Leave-one-subject-out cross-validation
    results = []
    for fold, test_name in enumerate(subject_names, start=1):
        print()
        print("======================================================")
        print(f"FOLD {fold}/{len(subject_names)}")
        print(f"Test subject: {test_name}")
        print("======================================================")

        remaining = [name for name in subject_names if name != test_name]

        # Deterministic validation selection: shuffle the remaining subjects with a
        # fold-specific seed and take the first N_VALIDATION_SUBJECTS. Their identity
        # changes with the held-out fold, but is reproducible.
        rng = random.Random(SEED + fold)
        shuffled_remaining = list(remaining)
        rng.shuffle(shuffled_remaining)

        validation_names = shuffled_remaining[:N_VALIDATION_SUBJECTS]
        training_names = [name for name in remaining if name not in validation_names]

        train_subjects = [subjects[name] for name in training_names]
        validation_subjects = [subjects[name] for name in validation_names]
        test_subject = subjects[test_name]

        print(f"Training subjects:   {len(train_subjects)}")
        print(f"Validation subjects: {len(validation_subjects)}")
        print(f"Test subjects:       1")

        model, scaler = train_fold(train_subjects, validation_subjects, test_subject, num_features)
        prediction, probabilities, weights = predict_subject(model, test_subject["graphs"], scaler)

        actual = test_subject["label"]
        correct = (prediction == actual)
        actual_name = "Alcoholic" if actual == 1 else "Control"
        predicted_name = "Alcoholic" if prediction == 1 else "Control"

        print()
        print(f"Subject:   {test_name}")
        print(f"Actual:    {actual_name}")
        print(f"Predicted: {predicted_name}")
        print(f"Trials:    {len(test_subject['graphs'])}")
        print(f"Correct:   {'YES' if correct else 'NO'}")
        print(f"P(Control):    {probabilities[0]:.4f}")
        print(f"P(Alcoholic):  {probabilities[1]:.4f}")

        # Which trials the model attended to most for this subject
        top_indices = np.argsort(weights)[::-1][:5]
        sorted_test_graphs = test_subject["graphs"]

        print()
        print("Top attended trials:")
        for index in top_indices:
            recording = sorted_test_graphs[index].recording
            print(f"    Trial {recording:3d} | Attention {weights[index]:.4f}")

        results.append({
            "fold": fold,
            "subject": test_name,
            "actual": actual,
            "prediction": prediction,
            "probability_control": float(probabilities[0]),
            "probability_alcoholic": float(probabilities[1]),
            "n_trials": len(test_subject["graphs"]),
            "correct": bool(correct),
            "attention_weights": weights.tolist(),
            "top_attention_trials": [
                {
                    "recording": int(sorted_test_graphs[index].recording),
                    "weight": float(weights[index]),
                }
                for index in top_indices
            ],
        })

    # Overall metrics across all folds
    y_true = np.array([result["actual"] for result in results])
    y_pred = np.array([result["prediction"] for result in results])

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print()
    print("======================================================")
    print("SUBJECT-LEVEL LOSO RESULTS")
    print("======================================================")
    print(f"Subjects:             {len(results)}")
    print(f"Accuracy:             {accuracy * 100:.2f}%")
    print(f"Balanced accuracy:    {balanced_accuracy * 100:.2f}%")
    print(f"Sensitivity:          {sensitivity * 100:.2f}%")
    print(f"Specificity:          {specificity * 100:.2f}%")

    print()
    print("Confusion matrix")
    print("                  Pred Control   Pred Alcoholic")
    print(f"Actual Control           {tn:2d}                {fp:2d}")
    print(f"Actual Alcoholic         {fn:2d}                {tp:2d}")

    # Per-subject breakdown
    print()
    print("Subject results:")
    for result in results:
        actual_name = "Alcoholic" if result["actual"] == 1 else "Control"
        predicted_name = "Alcoholic" if result["prediction"] == 1 else "Control"
        marker = "✓" if result["correct"] else "✗"
        print(
            f"  {marker} "
            f"{result['subject']} | "
            f"Actual={actual_name:9s} | "
            f"Predicted={predicted_name:9s} | "
            f"Trials={result['n_trials']:3d}"
        )

    # Save configuration, metrics and per-fold results (including attention weights)
    output = {
        "configuration": {
            "dataset": DATASET_FOLDER,
            "model": "GAT + learned trial attention",
            "node_features": 14,
            "connectivity": "Pearson top-8",
            "aggregation": "learned attention",
            "evaluation": "subject-level LOSO",
            "hidden_channels": HIDDEN_CHANNELS,
            "heads": HEADS,
            "attention_hidden": ATTENTION_HIDDEN,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "seed": SEED,
        },
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "results": results,
    }

    output_path = os.path.join(RESULTS_FOLDER, "gat_attention_loso_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("Results saved to:")
    print(output_path)
    print("======================================================")


if __name__ == "__main__":
    main()