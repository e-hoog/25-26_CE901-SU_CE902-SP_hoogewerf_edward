import os
import json
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Settings

# Change for dataset folder
DATASET_FOLDER = "data\\graphs"
RESULTS_FOLDER = "results"

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training

EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.30
HIDDEN_CHANNELS = 32
GAT_HEADS = 4

AGGREGATION = "mean"

# Early stopping
PATIENCE = 15

# Number of validation subjects
N_VALIDATION_SUBJECTS = 2

# A subject must have at least one valid trial
MIN_TRIALS_PER_SUBJECT = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Load dataset metadata (subject list, labels, feature config)

def load_metadata():
    metadata_path = os.path.join(DATASET_FOLDER, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find metadata:\n{metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load every saved trial graph belonging to one subject

def load_subject_graphs(subject, metadata):
    subject_info = metadata["subjects"][subject]
    graphs = []

    for recording in subject_info["trials"]:
        filename = f"{subject}_rd{recording:03d}.pt"
        path = os.path.join(DATASET_FOLDER, subject, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing graph:\n{path}")

        graph = torch.load(path, map_location="cpu", weights_only=False)
        graphs.append(graph)

    if len(graphs) < MIN_TRIALS_PER_SUBJECT:
        raise RuntimeError(f"Subject {subject} has only {len(graphs)} trials.")

    return graphs


# All subject ids present in the dataset

def get_subjects(metadata):
    subjects = sorted(metadata["subjects"].keys())
    if not subjects:
        raise RuntimeError("No subjects found.")
    return subjects


# Alcoholic/control label for one subject

def get_subject_label(subject, metadata):
    return int(metadata["subjects"][subject]["label"])


# Fit a StandardScaler using only the training subjects' node features, so nothing
# about validation or test subjects leaks into preprocessing

def fit_training_scaler(training_graphs):
    feature_arrays = [graph["x"].numpy() for graph in training_graphs]
    all_features = np.concatenate(feature_arrays, axis=0)

    scaler = StandardScaler()
    scaler.fit(all_features)
    return scaler


# Apply a fitted scaler to one graph's node features

def scale_graph(graph, scaler):
    x = graph["x"].numpy()
    original_shape = x.shape

    x_scaled = scaler.transform(x)
    if x_scaled.shape != original_shape:
        raise RuntimeError("Scaler changed graph shape.")

    graph_copy = {key: value for key, value in graph.items()}
    graph_copy["x"] = torch.tensor(x_scaled, dtype=torch.float32)
    graph_copy["scaled"] = True
    return graph_copy


# Apply a fitted scaler to a list of graphs

def scale_graphs(graphs, scaler):
    return [scale_graph(graph, scaler) for graph in graphs]


# Convert a saved graph dict into a PyTorch Geometric Data object

def to_pyg(graph):
    return Data(x=graph["x"], edge_index=graph["edge_index"], edge_weight=graph["edge_weight"])


class EEGGAT(nn.Module):

    def __init__(self, num_features, hidden_channels=32, heads=4, dropout=0.30):
        super().__init__()

        self.dropout = dropout

        # First graph-attention layer
        self.gat1 = GATConv(
            in_channels=num_features,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
        )

        first_output = hidden_channels * heads

        # Second graph-attention layer
        self.gat2 = GATConv(
            in_channels=first_output,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1,
        )

        # Subject classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

    # Run both GAT layers and pool nodes into one trial-level vector
    def trial_embedding(self, data):
        edge_weight = data.edge_weight.view(-1, 1)

        x = self.gat1(data.x, data.edge_index, edge_attr=edge_weight)
        x = torch.relu(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.gat2(x, data.edge_index, edge_attr=edge_weight)
        x = torch.relu(x)

        # [61 nodes, hidden] -> mean over nodes -> [hidden]
        embedding = x.mean(dim=0)
        return embedding

    # Combine a subject's trial embeddings into one subject-level vector
    def subject_embedding(self, trial_graphs):
        embeddings = [self.trial_embedding(graph) for graph in trial_graphs]
        embeddings = torch.stack(embeddings, dim=0)

        subject_embedding = embeddings.mean(dim=0)

        return subject_embedding

    # Full forward pass: a subject's trial graphs in, class logits out
    def forward_subject(self, trial_graphs):
        embedding = self.subject_embedding(trial_graphs)
        logits = self.classifier(embedding)
        return logits


# Convert a subject's graphs to PyG Data objects on the target device

def prepare_subject(graphs, device):
    return [to_pyg(graph).to(device) for graph in graphs]


# Train a GAT on the training subjects, using validation subjects for early stopping.
# Validation subjects are kept separate so a proper subject-level validation split
# can be used within each LOSO fold.

def train_fold(training_subjects, validation_subjects, metadata, num_features):
    # Load raw training graphs
    raw_training_graphs = []
    for subject in training_subjects:
        raw_training_graphs.extend(load_subject_graphs(subject, metadata))

    # Fit scaler ONLY on training subjects
    scaler = fit_training_scaler(raw_training_graphs)

    # Scale training subjects
    training_data = {}
    for subject in training_subjects:
        graphs = load_subject_graphs(subject, metadata)
        graphs = scale_graphs(graphs, scaler)
        training_data[subject] = prepare_subject(graphs, DEVICE)

    # Scale validation subjects
    validation_data = {}
    for subject in validation_subjects:
        graphs = load_subject_graphs(subject, metadata)
        graphs = scale_graphs(graphs, scaler)
        validation_data[subject] = prepare_subject(graphs, DEVICE)

    model = EEGGAT(
        num_features=num_features,
        hidden_channels=HIDDEN_CHANNELS,
        heads=GAT_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Early stopping state
    best_state = None
    best_validation_accuracy = -np.inf
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Randomise subject order every epoch
        shuffled_subjects = training_subjects.copy()
        random.shuffle(shuffled_subjects)

        training_predictions = []
        training_targets = []
        total_loss = 0.0

        for subject in shuffled_subjects:
            trial_graphs = training_data[subject]
            label = torch.tensor([get_subject_label(subject, metadata)], dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            logits = model.forward_subject(trial_graphs)
            loss = criterion(logits.unsqueeze(0), label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            prediction = int(logits.argmax().item())
            training_predictions.append(prediction)
            training_targets.append(int(label.item()))

        train_accuracy = accuracy_score(training_targets, training_predictions)

        # Validation
        model.eval()
        validation_predictions = []
        validation_targets = []

        with torch.no_grad():
            for subject in validation_subjects:
                trial_graphs = validation_data[subject]
                logits = model.forward_subject(trial_graphs)
                prediction = int(logits.argmax().item())
                target = get_subject_label(subject, metadata)

                validation_predictions.append(prediction)
                validation_targets.append(target)

        validation_accuracy = accuracy_score(validation_targets, validation_predictions)

        # Early stopping
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"    Epoch {epoch:3d}/{EPOCHS} | "
                f"Loss {total_loss / len(training_subjects):.4f} | "
                f"Train {train_accuracy * 100:6.2f}% | "
                f"Val {validation_accuracy * 100:6.2f}%"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler


# Run the trained model on one held-out test subject

def predict_subject(model, scaler, subject, metadata):
    model.eval()

    graphs = load_subject_graphs(subject, metadata)
    graphs = scale_graphs(graphs, scaler)
    graphs = prepare_subject(graphs, DEVICE)

    with torch.no_grad():
        logits = model.forward_subject(graphs)
        probabilities = torch.softmax(logits, dim=0)
        prediction = int(logits.argmax().item())

    target = get_subject_label(subject, metadata)

    return {
        "subject": subject,
        "target": target,
        "prediction": prediction,
        "probability_control": float(probabilities[0].item()),
        "probability_alcoholic": float(probabilities[1].item()),
        "n_trials": len(graphs),
    }


def run_loso():
    set_seed(SEED)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    print()
    print("======================================================")
    print("EEG GAT — SUBJECT-LEVEL LOSO")
    print("======================================================")
    print(f"Device:              {DEVICE}")
    print(f"Dataset:             {DATASET_FOLDER}")
    print(f"Aggregation:         {AGGREGATION}")
    print(f"Hidden channels:     {HIDDEN_CHANNELS}")
    print(f"GAT heads:           {GAT_HEADS}")
    print(f"Epochs:              {EPOCHS}")
    print()

    metadata = load_metadata()
    subjects = get_subjects(metadata)
    num_subjects = len(subjects)
    num_features = int(metadata["configuration"]["n_features"])

    print(f"Subjects:            {num_subjects}")
    print(f"Node features:       {num_features}")
    print()

    # Leave-one-subject-out cross-validation
    results = []
    for fold_index, test_subject in enumerate(subjects, start=1):
        print()
        print("======================================================")
        print(f"FOLD {fold_index}/{num_subjects}")
        print(f"Test subject: {test_subject}")
        print("======================================================")

        training_subjects = [subject for subject in subjects if subject != test_subject]

        # Deterministic validation selection: shuffle the remaining subjects with a
        # fold-specific seed and take the first N_VALIDATION_SUBJECTS. Their identity
        # changes with the held-out fold, but is reproducible.
        rng = random.Random(SEED + fold_index)
        shuffled_training_subjects = list(training_subjects)
        rng.shuffle(shuffled_training_subjects)

        validation_subjects = shuffled_training_subjects[:N_VALIDATION_SUBJECTS]
        actual_training_subjects = [
            subject for subject in training_subjects if subject not in validation_subjects
        ]

        # Leakage checks
        assert test_subject not in actual_training_subjects
        assert test_subject not in validation_subjects
        assert not (set(actual_training_subjects) & set(validation_subjects))

        print(f"Training subjects:   {len(actual_training_subjects)}")
        print(f"Validation subjects: {len(validation_subjects)}")
        print(f"Test subjects:       1")
        print()

        model, scaler = train_fold(actual_training_subjects, validation_subjects, metadata, num_features)
        result = predict_subject(model, scaler, test_subject, metadata)

        correct = (result["prediction"] == result["target"])
        result["fold"] = fold_index
        result["correct"] = bool(correct)
        results.append(result)

        actual_label = "Alcoholic" if result["target"] == 1 else "Control"
        predicted_label = "Alcoholic" if result["prediction"] == 1 else "Control"

        print()
        print(f"Subject:   {test_subject}")
        print(f"Actual:    {actual_label}")
        print(f"Predicted: {predicted_label}")
        print(f"Trials:    {result['n_trials']}")
        print(f"Correct:   {'YES' if correct else 'NO'}")
        print(f"P(Control):    {result['probability_control']:.4f}")
        print(f"P(Alcoholic):  {result['probability_alcoholic']:.4f}")

    # Overall metrics across all folds
    y_true = [result["target"] for result in results]
    y_pred = [result["prediction"] for result in results]

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
    print(f"Actual Control       {tn:5d}            {fp:5d}")
    print(f"Actual Alcoholic     {fn:5d}            {tp:5d}")

    # Per-subject breakdown
    print()
    print("Subject results:")
    for result in results:
        actual = "Alcoholic" if result["target"] == 1 else "Control"
        predicted = "Alcoholic" if result["prediction"] == 1 else "Control"
        marker = "✓" if result["correct"] else "✗"
        print(
            f"  {marker} "
            f"{result['subject']} | "
            f"Actual={actual:9s} | "
            f"Predicted={predicted:9s} | "
            f"Trials={result['n_trials']:3d}"
        )

    # Save configuration, metrics and per-fold results to disk
    output = {
        "configuration": {
            "dataset": DATASET_FOLDER,
            "aggregation": AGGREGATION,
            "hidden_channels": HIDDEN_CHANNELS,
            "gat_heads": GAT_HEADS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "seed": SEED,
            "device": str(DEVICE),
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
        "folds": results,
    }

    output_path = os.path.join(RESULTS_FOLDER, "gat_loso_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("Results saved to:")
    print(output_path)
    print("======================================================")


if __name__ == "__main__":
    run_loso()