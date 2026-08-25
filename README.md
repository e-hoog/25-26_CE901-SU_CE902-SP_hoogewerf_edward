# EEG-Based Alcoholism Classification Using Graph Attention Networks

## Note:
The final experiment results as displayed in the dissertation are in the final_results folder as to not be overwritten if the scripts are run.

## 1. Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Build dataset graphs

This pipeline is designed to work with the UCI EEG dataset. Download the large dataset here:
https://kdd.ics.uci.edu/databases/eeg/eeg.html
Then extract the files and place the subject folders in ./data/raw

Run:

```bash
python build_dataset.py
```

This should produce graph data used by training scripts.

## 3. Run experiments

### Subject-level GAT LOSO

```bash
python train_gat_loso.py
```

### Subject-level GAT + attention LOSO

```bash
python train_gat_attention_loso.py
```

### Classical baseline LOSO

```bash
python classical_baselines_loso.py
```

### Held-out trial GAT

```bash
python train_gat_trial_level.py
```

## 4. Outputs

- CSV/JSON outputs are written to `results/`.

## Notes

- If you already built the dataset, you can skip step 2.
- If a script fails due to missing data paths, check folder names and constants at the top of that script.
```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── build_dataset.py
├── train_gat_trial_level.py
├── train_gat_loso.py
├── train_gat_attention_loso.py
├── classical_baselines_loso.py
│
├── data/
│   ├── raw/                # place the downloaded UCI dataset here
│   └── graphs/              # graphs saved by build_dataset.py
│
├── final_results/           # results reported in the dissertation
│   ├── gat_trial_level_results.csv
│   ├── gat_loso_results.csv
│   ├── gat_attention_loso_results.csv
│   └── classical_baseline_results.csv
│
└── results/                  # output from re-running the scripts
```