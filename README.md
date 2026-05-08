# Diabetes Prediction MLP — PyTorch

Final project for Udacity's *Intro to PyTorch* nanodegree (nd101). A
binary classifier on the BRFSS 2015 Diabetes Health Indicators dataset
(~250 k rows, 21 features) built end-to-end in PyTorch.

## Notebook

```
starter-kit/diabetes_prediction_mlp.ipynb   solution notebook
starter-kit/data/diabetes_data.csv         input dataset
starter-kit/data/data_dictionary.md         column descriptions
```

## Pipeline

1. **EDA** — shape/dtypes, missing-value scan, target imbalance,
   summary statistics, continuous-feature distributions, target-correlation
   bar chart.
2. **Splitting + scaling** — stratified 70/15/15 train/val/test with
   `StandardScaler`, then converted to PyTorch `TensorDataset` +
   `DataLoader`.
3. **Model** — `DiabetesClassifier(in_features, hidden=(64,32))` MLP
   with ReLU + optional dropout, configurable hidden widths.
4. **Training** — `train_model(model, train_loader, val_loader, ...)` —
   per-epoch train + val loss tracking, returns history dict.
5. **Evaluation** — `evaluate_model(model, loader)` returns accuracy,
   precision, recall, F1, plus confusion matrix and ROC/AUC plots.
6. **Experiments** —
   * dropout=0.3 baseline,
   * learning-rate sweep (1e-4, 1e-3, 1e-2),
   * "best" model: 128 → 64 → 32 hidden, dropout=0.3,
     `AdamW(lr=1e-3, weight_decay=1e-4)`, 30 epochs.

Each experiment writes into a shared `experiment_results` dict for the
final comparison cells.

## Running

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
jupyter notebook starter-kit/diabetes_prediction_mlp.ipynb
# Restart & Run All
```

## Standing-out work

* **Configurable architecture** — `hidden=(...)` accepts arbitrarily
  many hidden widths so the lr-sweep and the best model use the
  same class.
* **Single training loop** for every experiment — the comparison
  table in the notebook reads from the same `evaluate_model` output
  shape regardless of experiment.
* **AdamW + weight decay** in the best-config experiment instead of
  vanilla Adam — matches modern best practice for tabular MLPs.

## License

Educational submission for Udacity nd101. Dataset © CDC BRFSS 2015.
