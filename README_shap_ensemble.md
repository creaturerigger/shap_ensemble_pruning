# SHAP-Based Ensemble Pruning for Vision Models

This repository provides an implementation of an ensemble pruning strategy using **SHAP values** for vision models trained on datasets like MNIST and CIFAR-10. The pruning method leverages interpretable pixel-wise importance to select robust and performant subsets of convolutional neural networks (CNNs).

> 🛠️ This project is under active development. Expect improvements to pruning algorithms, logging, and visual diagnostics.

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20MacOS%20%7C%20Windows-lightgrey)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

---

## 📋 Summary

The goal of this project is to prune large ensembles of CNNs by analysing their SHAP-based explanations. The selected models are those whose importance maps suggest diverse and non-redundant reasoning paths. Evaluation is done both on masked test sets and original datasets.

Core features include:

- Training CNNs with masked data using SHAP explanations.
- SHAP-guided pruning to select models contributing complementary knowledge.
- SIP (Semi-Infinite Programming)-based soft pruning with learnable weights.
- Evaluation pipeline to benchmark pruning performance across top-k thresholds.

---

## 🔧 Setup

### Requirements

Install using `pip`:

```bash
pip install -r requirements.txt
```

Or if you use `uv`:

```bash
uv venv
uv pip install -e .
```

### Directory Structure

```
ensemble_models/
├── with_shap/
│   └── MODEL*/  # Contains saved models and their prediction files
logs/
src/
├── pruning/
│   └── sip_pruning.py
├── trainer/
│   └── cnn_trainer.py
```

---

## 🧪 Training & Pruning Flow

To run the training, pruning, and evaluation pipeline:

```bash
python scripts/training_flow.py
```

This will:

1. Train an ensemble of CNNs.
2. Generate SHAP explanations.
3. Apply pruning using SHAP similarity and SIP optimization.
4. Evaluate on both masked and original test sets.
5. Save results under `logs/shap_values_ensemble_logs/`.

---

## 📊 Evaluation Metrics

| Metric             | Description                                  |
|--------------------|----------------------------------------------|
| Accuracy           | Overall correct predictions                  |
| Precision (macro)  | Average precision across all classes         |
| Recall (macro)     | Average recall across all classes            |
| F1 Score (macro)   | Harmonic mean of macro precision and recall  |

Results are reported for both:
- Ensemble of **masked data** predictions.
- Ensemble of **original data** predictions (post-pruning).

---

## 📁 Logs & Results

Each run will output:

```bash
logs/shap_values_ensemble_logs/
└── ensemble_pruning_results_{dataset_name}.csv
```

---

## 📚 Citation

If this repository contributed to your work, please cite:

```bibtex
@misc{shapensemble2025,
  title        = {SHAP-Based Pruning of Deep Ensembles for Vision Tasks},
  author       = {Volkan Bakir},
  year         = {2025},
  howpublished = {\url{https://github.com/volkanbakir/shap-ensemble-pruning}},
  note         = {Work in Progress}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.