import os
import gzip
import re
import json

import numpy as np
import torch

from scipy.signal import welch, butter, sosfiltfilt

# Settings

# Change for dataset folder
RAW_FOLDER = os.path.join("data", "raw")
OUTPUT_FOLDER = os.path.join("data", "graphs")

FS = 256.0

N_CHANNELS = 61
N_SAMPLES = 256

TOP_K = 8

# Order of the band-pass filter used for band power features
FILTER_ORDER = 4

ZERO_VARIANCE_THRESHOLD = 1e-12

NON_EEG_CHANNELS = {
    "X",
    "Y",
    "nd"
}

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# Feature names

FEATURE_NAMES = (
    [f"{band_name}_power" for band_name in BANDS]
    + [f"{band_name}_relative" for band_name in BANDS]
    + [
        "spectral_entropy",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity"
    ]
)

N_FEATURES = len(FEATURE_NAMES)


# Parse subject id and channel names from a .rd header

def parse_header(lines):
    subject = None
    channel_names = []
    for line in lines:
        line = line.strip()
        # Example:
        #
        # # co2a0000364.rd
        if line.startswith("# co2"):
            subject = line[2:].strip()
            if subject.endswith(".rd"):
                subject = subject[:-3]
        elif line.startswith("#") and "chan" in line:
            parts = line[2:].split()
            if len(parts) >= 3 and parts[1] == "chan":
                channel_names.append(parts[0])
    if subject is None:
        raise ValueError("Could not determine subject")
    return subject, channel_names


# Alcoholic/control label from the subject code

def get_label(subject):
    code = subject[3].lower()
    if code == "a":
        return 1
    if code == "c":
        return 0
    raise ValueError(f"Unknown subject label: {subject}")


# Read and validate one EEG .gz trial file

def read_gz_file(filename):
    # Read the whole file so header and samples can be validated together
    with gzip.open(filename, "rt") as f:
        lines = f.readlines()

    subject, channel_names = parse_header(lines)

    if len(channel_names) != 64:
        raise ValueError(f"{subject}: expected 64 channels, found {len(channel_names)}")

    eeg_channels = [ch for ch in channel_names if ch not in NON_EEG_CHANNELS]

    if len(eeg_channels) != N_CHANNELS:
        raise ValueError(f"{subject}: expected {N_CHANNELS} EEG channels, found {len(eeg_channels)}")

    channel_to_index = {ch: i for i, ch in enumerate(eeg_channels)}
    source_filename = os.path.basename(filename)
    match = re.search(r"\.rd\.(\d+)\.gz$", source_filename)

    if match is None:
        raise ValueError(f"{subject}: could not determine recording number from {source_filename}")

    recording_number = int(match.group(1))

    data = np.zeros((N_CHANNELS, N_SAMPLES), dtype=np.float32)
    sample_counts = np.zeros(N_CHANNELS, dtype=np.int32)

    # Keep only valid EEG samples and place them in channel order
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            channel = parts[1]
            sample = int(parts[2])
            value = float(parts[3])
        except ValueError:
            continue
        if channel not in channel_to_index:
            continue
        if not (0 <= sample < N_SAMPLES):
            continue
        node = channel_to_index[channel]
        data[node, sample] = value
        sample_counts[node] += 1

    if np.any(sample_counts != N_SAMPLES):
        bad = np.where(sample_counts != N_SAMPLES)[0]
        names = [eeg_channels[i] for i in bad]
        raise ValueError(f"{subject}: recording {recording_number}: incorrect sample counts for {names}")

    return (subject, get_label(subject), eeg_channels, recording_number, source_filename, data)


# Find .gz files in the raw dataset folder

def find_gz_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(".gz"):
                files.append(os.path.join(root, filename))
    return sorted(files)


# Unique id for a subject's trial, used to catch duplicates

def trial_key(subject, recording_number):
    return f"{subject}|" f"rd{recording_number:03d}"


# Zero-phase (forward/backward) band-pass filter, applied per channel

def bandpass_filter(data, low, high, fs, order=FILTER_ORDER):
    nyquist = fs / 2.0
    sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=1)


# Band powers via forward/backward band-pass filtering

def calculate_band_power_features(data):
    absolute_features = []

    for low, high in BANDS.values():
        filtered = bandpass_filter(data, low, high, FS)
        power = np.mean(filtered ** 2, axis=1)
        absolute_features.append(power)

    absolute = np.stack(absolute_features, axis=1)

    total_power = np.sum(absolute, axis=1, keepdims=True)
    relative = absolute / (total_power + 1e-12)

    return absolute, relative


# Spectral entropy from Welch's method

def calculate_spectral_entropy(data):
    frequencies, psd = welch(data, fs=FS, axis=1, nperseg=min(128, data.shape[1]))

    total_mask = (frequencies >= 1) & (frequencies < 45)
    psd_selected = psd[:, total_mask]
    probability = psd_selected / (np.sum(psd_selected, axis=1, keepdims=True) + 1e-12)

    entropy = -np.sum(probability * np.log(probability + 1e-12), axis=1)
    return entropy[:, None]


def calculate_spectral_features(data):
    absolute, relative = calculate_band_power_features(data)
    entropy = calculate_spectral_entropy(data)
    return np.concatenate([absolute, relative, entropy], axis=1)


# Extract Hjorth features from EEG

def calculate_hjorth_features(data):
    activity = np.var(data, axis=1)

    d1 = np.diff(data, axis=1)
    d2 = np.diff(d1, axis=1)

    var_d1 = np.var(d1, axis=1)
    var_d2 = np.var(d2, axis=1)

    mobility = np.sqrt(var_d1 / (activity + 1e-12))
    mobility_d1 = np.sqrt(var_d2 / (var_d1 + 1e-12))

    complexity = mobility_d1 / (mobility + 1e-12)

    return np.stack([activity, mobility, complexity], axis=1)


def calculate_node_features(data):
    spectral = calculate_spectral_features(data)
    hjorth = calculate_hjorth_features(data)
    x = np.concatenate([spectral, hjorth], axis=1)

    if x.shape != (N_CHANNELS, N_FEATURES):
        raise RuntimeError(
            f"Unexpected feature shape: {x.shape}; expected ({N_CHANNELS}, {N_FEATURES})")
    return x.astype(np.float32)


# Calculate Pearson correlation graph from EEG

def calculate_pearson_graph(data):
    # Each channel keeps its TOP_K strongest correlations by magnitude as directed edges
    correlation = np.corrcoef(data)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)
    correlation_magnitude = np.abs(correlation)

    sources = []
    targets = []
    weights = []

    for source in range(N_CHANNELS):
        row = correlation_magnitude[source].copy()
        row[source] = -np.inf
        neighbours = np.argpartition(row, -TOP_K)[-TOP_K:]
        for target in neighbours:
            sources.append(source)
            targets.append(target)
            weights.append(correlation_magnitude[source, target])

    edge_index = np.array([sources, targets], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    return (edge_index, edge_weight)


# Process every raw file into a saved graph

def process_all(files):
    processed = 0
    skipped = 0
    trial_keys = set()
    subjects = {}
    metadata = []
    for number, gz_file in enumerate(files, start=1):
        print(f"[{number}/{len(files)}] " f"{os.path.basename(gz_file)}")
        try:
            subject, label, channel_names, recording_number, source_filename, data = read_gz_file(gz_file)
            key = trial_key(subject, recording_number)
            if key in trial_keys:
                raise ValueError(f"Duplicate trial: {key}")
            trial_keys.add(key)

            # Check if channels have zero variance
            channel_std = np.std(data, axis=1)
            if np.any(channel_std < ZERO_VARIANCE_THRESHOLD):
                bad = np.where(channel_std < ZERO_VARIANCE_THRESHOLD)[0]
                names = [channel_names[i] for i in bad]
                raise ValueError(f"zero-variance " f"channels: {names}")

            x = calculate_node_features(data)

            edge_index, edge_weight = calculate_pearson_graph(data)

            assert x.shape == (N_CHANNELS, N_FEATURES)
            assert edge_index.shape == (2, N_CHANNELS * TOP_K)
            assert edge_weight.shape == (N_CHANNELS * TOP_K,)
            assert np.isfinite(x).all()
            assert np.isfinite(edge_weight).all()

            if subject not in subjects:
                subjects[subject] = {
                    "label": label,
                    "trials": []
                }
            else:
                if subjects[subject]["label"] != label:
                    raise RuntimeError(f"Label inconsistency for subject {subject}")

            graph = {
                "x": torch.tensor(x, dtype=torch.float32),
                "edge_index": torch.tensor(edge_index, dtype=torch.long),
                "edge_weight": torch.tensor(edge_weight, dtype=torch.float32),
                "y": torch.tensor(label, dtype=torch.long),
                "subject": subject,
                "recording": recording_number,
                "source_file": source_filename,
                "channel_names": channel_names,
                "connectivity": "abs_Pearson",
                "top_k": TOP_K,
                "features": FEATURE_NAMES,
                "trial_key": key,
                "scaled": False
            }

            subject_folder = os.path.join(OUTPUT_FOLDER, subject)
            os.makedirs(subject_folder, exist_ok=True)
            output_name = f"{subject}_" f"rd{recording_number:03d}.pt"
            output_file = os.path.join(subject_folder, output_name)
            if os.path.exists(output_file):
                raise FileExistsError(f"OUTPUT ALREADY EXISTS: " f"{output_file}")
            torch.save(graph, output_file)
            subjects[subject]["trials"].append(recording_number)
            metadata.append(
                {
                    "trial_key": key,
                    "subject": subject,
                    "recording": recording_number,
                    "label": label,
                    "source_file": source_filename
                }
            )
            processed += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            skipped += 1
    return {
        "processed": processed,
        "skipped": skipped,
        "trial_keys": trial_keys,
        "subjects": subjects,
        "metadata": metadata,
    }


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    files = find_gz_files(RAW_FOLDER)
    print()
    print("======================================================")
    print("LOSO SUBJECT-LEVEL GAT GRAPH BUILDER")
    print("======================================================")
    print(f"Raw files:       {len(files)}")
    print(f"Subjects:        determined from data")
    print(f"Nodes:           {N_CHANNELS}")
    print(f"Node features:   {N_FEATURES}")
    print(f"Connectivity:    abs_Pearson")
    print(f"Top-K:           {TOP_K}")
    print()
    if not files:
        raise RuntimeError(f"No .gz files found in " f"{RAW_FOLDER}")
    result = process_all(files)
    subjects = result["subjects"]

    # Validate subjects
    alcoholic = sorted(subject for subject, info in subjects.items() if info["label"] == 1)
    control = sorted(subject for subject, info in subjects.items() if info["label"] == 0)
    # Every subject must have at least one trial
    for subject, info in subjects.items():
        if not info["trials"]:
            raise RuntimeError(f"Subject {subject} " f"has no trials")

    # Write metadata
    metadata = {
        "configuration": {
            "fs": FS,
            "n_channels": N_CHANNELS,
            "n_samples": N_SAMPLES,
            "top_k": TOP_K,
            "connectivity": "abs_Pearson",
            "features": FEATURE_NAMES,
            "n_features": N_FEATURES,
        },
        "subjects": {
            subject: {
                "label": info["label"],
                "n_trials": len(info["trials"]),
                "trials": sorted(info["trials"])
            }
            for subject, info in sorted(subjects.items())
        },
        "trials": result["metadata"]
    }
    metadata_path = os.path.join(OUTPUT_FOLDER, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Print final summary
    print()
    print("======================================================")
    print("DATASET BUILD COMPLETE")
    print("======================================================")
    print(f"Graphs created:    " f"{result['processed']}")
    print(f"Files skipped:     " f"{result['skipped']}")
    print(f"Unique subjects:   " f"{len(subjects)}")
    print(f"Alcoholic:         " f"{len(alcoholic)}")
    print(f"Control:           " f"{len(control)}")
    print()
    print(f"Features ({N_FEATURES}):")
    for index, name in enumerate(FEATURE_NAMES, start=1):
        print(f"  {index:2d}. {name}")
    print()
    print("Subject trial counts:")
    for subject in sorted(subjects):
        info = subjects[subject]
        label_name = "Alcoholic" if info["label"] == 1 else "Control"
        print(f"  {subject}: " f"{label_name:9s} " f"{len(info['trials']):3d} trials")
    print()
    print(f"Output:   {OUTPUT_FOLDER}")
    print(f"Metadata: {metadata_path}")
    print("======================================================")


if __name__ == "__main__":
    main()