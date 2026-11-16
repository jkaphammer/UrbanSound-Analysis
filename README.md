# UrbanSound8K: noise-event triage classifier

**Decision:** Flag audio segments that may contain drilling or jackhammer sounds for review by municipal staff, **not** automated enforcement. False positives consume staff review time; false negatives miss relevant events. Primary metric: **F0.5**, emphasizing precision.

**Reproduced headline:** unweighted mean F0.5 over the ten *original* UrbanSound8K held-out folds: **L2 logistic regression 0.933389**, **random forest 0.935408**. Declared baselines on the same metric/splits: **all-positive 0.554009**, **loudness-only approximately 0.6744**. The recommendation favors no-PCA logistic regression as the simpler candidate pending field validation.

## Reproduce the headline from a fresh clone

Run these commands **from the repository root** in a terminal with Python 3.11+ installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
```

**Windows PowerShell:** activate with `.venv\Scripts\Activate.ps1` and use `python` instead of `python3` to create the environment, if necessary.

The exact reproducibility command is **`python run_analysis.py`**. It loads `data/urban_features.csv`, evaluates both baselines and both trained models (plus the PCA experiment), prints all four model scores, asserts the saved headline scores within `1e-5`, and saves the executed notebook as `UrbanSound_reproduced.ipynb`. **Nested cross-validation is computationally expensive**; this script does not download the large WAV collection.

Quick imports/data/first-figure sanity check without running model evaluation:

```bash
python run_analysis.py --smoke
```

## Files and provenance

- `UrbanSound_Reviewed_Visualized.ipynb`: complete analysis, seven plots, subgroup confidence intervals, decision discussion, and an editable AI-use appendix draft.
- `data/urban_features.csv`: **included** extracted feature table (4,000 rows, 55 columns); no network access required for model evaluation.
- `run_analysis.py`: executes the notebook from imports onwards while **skipping Section 1**, which is an optional extraction workflow requiring approximately 6 GB of audio files.
- `requirements.txt`: Python packages required by notebook and runner.

The features were produced from the UrbanSound8K dataset (four classes: children_playing, street_music, drilling, jackhammer; two labels) using `librosa`, with per-clip acoustic summaries and the dataset's original folds. The optional notebook Section 1 shows raw-audio processing and uses KaggleHub, but the included extracted CSV is the source for the evaluation results.

## How the evaluation works

- Original ten UrbanSound8K folds are kept intact: 9 outer training folds and 1 held-out fold per iteration.
- Within each outer training partition, `LeaveOneGroupOut` chooses hyperparameters and classification threshold using the remaining fold IDs.
- Scaling and PCA are fit inside the training partitions, not on test recordings.
- Main score is **mean outer-fold F0.5**, whereas R5 subgroup F0.5 pools the held-out predictions within each subgroup.
- Fixed randomness uses `random_state=42` for models and seed 42 for fold-bootstrap intervals; bootstrap uncertainty is approximate because it resamples only ten folds.

## Important limitations

The CSV is derived data; it cannot recreate original WAV waveforms or spectrograms. For raw audio, execute Section 1 manually and ensure sufficient space. The four-class experiment is not an evaluation of open-world city audio. Loudness/duration subgroup class balances differ considerably; the observed worst subgroup is exploratory.