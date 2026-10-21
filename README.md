# PGNN-ASL: Physics-Guided Neural Network for Subsurface Damage Prediction

**Interpretable Subsurface Damage Prediction for Optical Components via Physics-Guided Neural Network Adaptive Symbolic Layer**

This repository contains the research code for predicting subsurface damage (SSD) depth in hard-brittle optical materials during precision grinding. The project combines physics-informed feature engineering, adaptive symbolic learning, neural-network residual correction, and uncertainty estimation to improve predictive accuracy and physical interpretability.

## 1. Overview

Subsurface damage is an important factor affecting the surface integrity, optical performance, and subsequent material-removal requirements of precision optical components. Conventional data-driven regression models can capture nonlinear relationships between grinding parameters and SSD depth, but their predictions may be difficult to interpret physically.

To address this challenge, this project develops a **Physics-Guided Neural Network with an Adaptive Symbolic Layer (PGNN-ASL)**.

The framework integrates physical descriptors derived from fracture mechanics with a differentiable symbolic learning mechanism. A residual neural branch complements the symbolic branch to model nonlinear effects that cannot be adequately represented by a compact analytical expression.

The main objectives are:

- Predict SSD depth from material properties and grinding conditions.
- Incorporate fracture-mechanics-inspired descriptors into the learning process.
- Discover interpretable symbolic relationships associated with SSD formation.
- Estimate predictive uncertainty to support risk-aware manufacturing decisions.
- Evaluate the contributions of individual model components through ablation studies.
- Compare the proposed model against conventional machine-learning baselines.

## 2. Methodology

### 2.1 Input features

The six principal physical input variables are:

| Symbol | Description | Dataset column |
|---|---|---|
| \(H\) | Mohs hardness | `莫氏硬度` |
| \(E\) | Elastic modulus | `弹性模量` |
| \(K_c\) | Fracture toughness | `断裂韧性` |
| \(d\) | Abrasive grain size | `晶粒粒径` |
| \(\sigma\) | Equivalent grinding pressure | `压溃压力` |
| \(v\) | Abrasive grain velocity | `磨粒速度` |

The target variable is the maximum subsurface crack depth, stored in `亚表面裂纹深度`.

### 2.2 Physics-informed feature extraction

The physics-informed feature extractor combines the six original variables with twelve additional descriptors, producing an 18-dimensional physical feature representation.

The derived descriptors represent different aspects of SSD formation, including:

- Griffith energy release rate.
- Stress-intensity ratio.
- Grain-size scaling term.
- Crack-shielding factor.
- Load-strength ratio.
- Power-strength ratio.
- Critical brittle-to-ductile transition depth.
- Critical crack ratio.
- Plastic-zone ratio.
- Material brittleness index.
- Toughness-to-modulus ratio.
- Elastic strain energy density.

These descriptors provide physically traceable representations of crack-driving energy, material resistance, loading conditions, and geometric scale effects.

### 2.3 Adaptive symbolic learning

The adaptive symbolic layer (ASL) combines physical information with a learned latent representation.

Its main components include:

1. **Cross-attention:** connects latent representations with physics-informed features.
2. **Variable selection:** learns the relative importance of candidate physical descriptors.
3. **Operator selection:** learns combinations of mathematical operations.
4. **Coefficient estimation:** generates coefficients for symbolic terms.
5. **Symbolic gating:** balances different symbolic components.

The symbolic branch provides a compact, interpretable representation, while the residual neural branch captures additional nonlinear dependencies.

### 2.4 Gated fusion and uncertainty estimation

The symbolic and neural branches are combined through a learnable gating mechanism:

\[
\hat{y}=\alpha y_{\mathrm{sym}}+(1-\alpha)y_{\mathrm{nn}},
\]

where \(y_{\mathrm{sym}}\) and \(y_{\mathrm{nn}}\) denote the symbolic and neural predictions, respectively.

The model also incorporates uncertainty-related outputs. Monte Carlo dropout is supported by the model implementation for uncertainty estimation.

### 2.5 Physics-guided optimization

The training framework incorporates prediction fitting, physical regularization, symbolic complexity control, and gating regularization.

The physical constraints provide soft guidance rather than universally valid hard constraints. This allows the network to accommodate deviations from simplified fracture-mechanics relationships.

## 3. Architecture

The overall workflow can be summarized as follows:

```text
Material Properties and Grinding Parameters
                    |
                    v
       Six-Dimensional Input Features
                    |
                    v
     Physics-Informed Feature Extractor
       6 Raw + 12 Derived Features
                    |
                    v
          Latent Representation
                    |
          +---------+---------+
          |                   |
          v                   v
   Adaptive Symbolic      Residual Neural
        Branch                Branch
          |                   |
          +---------+---------+
                    |
                    v
              Gated Fusion
                    |
                    v
       SSD Depth and Uncertainty
                    |
                    v
       Evaluation and Interpretation
```

The repository implementation also includes material-category embeddings in `new_pgnn/models.py`. The exact implementation details should be checked against the intended model version when reproducing individual paper experiments.

## 4. Repository Structure

```text
pgnn_srnet_1/
├── 3011916164精确材料性能数据.csv
├── new_pgnn/
│   ├── models.py
│   ├── models_ablation.py
│   ├── losses.py
│   ├── utils.py
│   ├── train.py
│   ├── train_hyperopt.py
│   ├── test.py
│   ├── model_pre.py
│   ├── data_prs.py
│   ├── ImprovedSymbolicLayer.py
│   ├── light_symbolic_layer.py
│   ├── symbolic_decoder.py
│   ├── ablation_study.py
│   ├── interpretability_analysis.py
│   ├── generate_attention_plot.py
│   ├── generate_uncertainty_plot.py
│   ├── generate_ablation_comparison_plot.py
│   └── show.py
├── old_pgnn/
│   ├── models.py
│   ├── losses.py
│   ├── utils.py
│   ├── train.py
│   └── train_hyperopt.py
├── improved_material_crack_pgnn.py
├── improved_pgnn_hyperopt_r方0.7.py
├── fair_model_comparison.py
├── comparison_with_hyperopt.py
├── improved_physics_features_analysis.py
├── material_data_analysis.py
├── equation.py
├── test.py
└── .gitignore
```

### Core modules

| File | Purpose |
|---|---|
| `new_pgnn/models.py` | Main model implementation, physics-informed features, symbolic learning, residual blocks, gating, and uncertainty-related interfaces. |
| `new_pgnn/losses.py` | Physics-informed loss functions and regularization. |
| `new_pgnn/utils.py` | Data preprocessing, material encoding, normalization, and data augmentation utilities. |
| `new_pgnn/train.py` | Main fixed-hyperparameter training and evaluation workflow. |
| `new_pgnn/train_hyperopt.py` | Optuna-based hyperparameter optimization. |
| `new_pgnn/test.py` | Extended training/evaluation workflow, including cross-validation options and interpretability analysis. |
| `new_pgnn/model_pre.py` | `CrackDepthPredictor` interface for CSV-based and single-sample predictions. |
| `new_pgnn/symbolic_decoder.py` | Converts learned symbolic-layer representations into interpretable mathematical expressions. |
| `new_pgnn/interpretability_analysis.py` | Feature-importance, mechanism-attention, symbolic-topology, and coefficient analyses. |
| `new_pgnn/models_ablation.py` | Model variants for ablation experiments. |
| `new_pgnn/ablation_study.py` | Ablation training, evaluation, comparison tables, and visualizations. |
| `new_pgnn/generate_attention_plot.py` | Generates physical-feature attention visualizations. |
| `new_pgnn/generate_uncertainty_plot.py` | Generates prediction and uncertainty plots using a trained checkpoint. |
| `new_pgnn/generate_ablation_comparison_plot.py` | Generates ablation comparison figures. |
| `new_pgnn/data_prs.py` | Converts grinding process parameters into model-related input quantities. |
| `old_pgnn/` | Earlier model, training, loss, preprocessing, and hyperparameter-search implementations. |
| `fair_model_comparison.py` | Compares PGNN with traditional regression and ensemble-learning baselines. |
| `comparison_with_hyperopt.py` | Additional model-comparison workflow involving hyperparameter optimization. |
| `improved_material_crack_pgnn.py` | Earlier standalone implementation of the material-crack prediction model. |
| `improved_pgnn_hyperopt_r方0.7.py` | Earlier model implementation and hyperparameter optimization workflow. |
| `material_data_analysis.py` | Exploratory material-data analysis. |
| `improved_physics_features_analysis.py` | Physics-informed feature analysis. |
| `equation.py` | Evaluates the extracted symbolic surrogate relationship. |
| `test.py` | Root-level experiment script associated with the older implementation. |

The `new_pgnn/` directory is the recommended starting point for understanding the more recent model implementation. The root-level and `old_pgnn/` scripts preserve additional development and experimental workflows.

## 5. Experimental Results

The following results are reported in the accompanying paper. They are reference results from the manuscript, not a guarantee that every script or configuration in this repository will reproduce them automatically.

### 5.1 Comparison with conventional models

| Model | RMSE (μm) ↓ | MAE (μm) ↓ | \(R^2\) ↑ |
|---|---:|---:|---:|
| Gradient Boosting | 1.459 | 1.054 | 0.6741 |
| XGBoost | 1.432 | 1.077 | 0.6857 |
| SVR | 1.368 | 1.014 | 0.7132 |
| ElasticNet | 1.356 | 1.038 | 0.7182 |
| Random Forest | 1.354 | 0.991 | 0.7191 |
| **PGNN-ASL** | **1.252** | **0.926** | **0.7600** |

Compared with Random Forest, PGNN-ASL reduces RMSE by approximately 7.5% and MAE by approximately 6.6%.

### 5.2 Ablation study

| Model variant | RMSE (μm) ↓ | MAE (μm) ↓ | \(R^2\) ↑ |
|---|---:|---:|---:|
| PGNN without physics-informed features | 1.297 | 0.965 | 0.7423 |
| PGNN without ASL | 1.317 | 1.007 | 0.7342 |
| **Full PGNN-ASL** | **1.252** | **0.926** | **0.7600** |

The reported ablation results indicate that physics-informed features and the adaptive symbolic layer make complementary contributions to prediction performance.

### 5.3 Experimental validation

The paper further evaluates the model on glass-ceramic specimens under previously unseen material and grinding conditions.

Subsurface damage depth is measured using the magnetorheological finishing (MRF) spot method. The manuscript reports good agreement between predicted and measured SSD depths, with the measured values falling within the predicted confidence intervals for the reported validation samples.

## 6. Environment and Dependencies

The paper reports an experimental environment based on:

- Python deep-learning framework: PyTorch 2.2.0.
- CUDA 12.1.
- CPU: Intel Core i7-14650HX.
- GPU: NVIDIA RTX 4070, 16 GB VRAM.

The repository does not currently provide a unified `requirements.txt`. The following setup is a starting point rather than a verified, fully pinned dependency specification.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the general dependencies:

```bash
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib optuna joblib sympy seaborn
```

Install PyTorch separately using the official installation instructions appropriate for your operating system and CUDA configuration. The paper's reported environment used PyTorch 2.2.0 with CUDA 12.1.

For the traditional-model comparison workflow, XGBoost is optional:

```bash
pip install xgboost
```

A CUDA-capable GPU is recommended for training, although CPU execution may be possible depending on the experiment and available resources.

## 7. Dataset

The repository includes the dataset:

```text
3011916164精确材料性能数据.csv
```

The training scripts use the following named columns:

```text
材料种类
莫氏硬度
弹性模量
断裂韧性
晶粒粒径
压溃压力
磨粒速度
亚表面裂纹深度
```

The CSV also contains a sample identifier, `样本编号`.

The model uses the six numerical physical variables as its principal predictors, while `亚表面裂纹深度` is the regression target. The preprocessing utilities map the columns by name, so the CSV column order should not be changed without checking the preprocessing code.

Important data-handling considerations:

- Keep the input feature definitions consistent with the training configuration.
- Ensure that all numerical values use units consistent with the training dataset.
- Fit normalization and material encoders on the training data and reuse them for validation, testing, and inference.
- Do not include target information in model inputs.
- Keep training and test partitions consistent when comparing model variants.

The manuscript describes a dataset of approximately 1,000 observations and an 80:20 training/test split stratified by material category. Reproduction of the reported results also depends on the exact data version, random seed, preprocessing, and model configuration.

## 8. Training

### 8.1 Main training workflow

Run commands from the repository root so that the relative dataset path resolves correctly:

```bash
python new_pgnn/train.py
```

The script defines fixed hyperparameters and performs training and evaluation. The main configuration includes:

- Material embedding dimension.
- Latent dimension.
- Dropout rate.
- Symbolic-layer type.
- Learning rate and optimizer.
- Batch size and training epochs.
- Early stopping.
- Physics and symbolic-complexity regularization.
- Optional data augmentation.

To change the experimental setup, edit the `PARAMS` dictionary and relevant configuration constants in `new_pgnn/train.py`.

### 8.2 Hyperparameter optimization

The repository includes an Optuna-based hyperparameter search:

```bash
python new_pgnn/train_hyperopt.py
```

**Before running this script, check its dataset path.** The current default refers to `new_pgnn/3011916164精确材料性能数据.csv`, while the supplied CSV is located in the repository root. Update the default path or the `run_hyperopt()` call to use:

```python
data_path="3011916164精确材料性能数据.csv"
```

The search workflow supports optimization of model architecture and training parameters and can save study results and trained checkpoints.

## 9. Evaluation and Additional Experiments

### Traditional-model comparison

```bash
python fair_model_comparison.py
```

This script compares PGNN with conventional machine-learning models, including Random Forest, Gradient Boosting, SVR, ElasticNet, and optionally XGBoost.

### Physics-feature analysis

```bash
python improved_physics_features_analysis.py
```

### Symbolic surrogate evaluation

```bash
python equation.py
```

### Ablation study

```bash
python new_pgnn/ablation_study.py
```

The ablation workflow evaluates variants that remove or modify components such as physics-informed features, symbolic learning, residual learning, and gating.

Some analysis and plotting scripts use hard-coded checkpoint paths. Verify the configured model path and result directory before running them.

## 10. Prediction and Interpretability

### Prediction on new data

The `new_pgnn/model_pre.py` module provides a `CrackDepthPredictor` interface for CSV-based and single-sample prediction.

Before using it, configure the following paths in the script:

- Trained model checkpoint.
- Training dataset used to recover preprocessing information.
- New input CSV.
- Prediction output CSV.
- Optional visualization output directory.

The current script contains example paths that may not match the checkpoint produced by the main training script.

The repository also includes tools for:

- Physical-feature attention visualization.
- Symbolic expression decoding.
- Symbolic operator and coefficient analysis.
- Prediction-versus-measurement plots.
- Uncertainty visualization.
- Ablation comparison figures.

These analyses help relate the learned prediction behavior to physical descriptors and provide additional information beyond point-prediction accuracy.

## 11. Output Files

The training and experiment scripts save results under `results/`, often in timestamped subdirectories.

Depending on the workflow, outputs may include:

```text
results/
└── run_<timestamp>/
    ├── train_model.pth
    ├── training_plots.png
    ├── gate_analysis.png
    ├── predictions_visualization.png
    ├── uncertainty_visualization.png
    ├── prediction_results.csv
    └── run_summary.txt
```

Not every file is generated by every script. Check the console output and the specific script implementation for the actual checkpoint name and output paths.

## 12. Reproducibility Notes

This repository contains multiple iterations of the model and experimental scripts. The main implementation under `new_pgnn/` should be distinguished from the earlier implementations under `old_pgnn/` and the root directory.

For reliable reproduction:

1. Record the code version, random seed, dataset version, and hyperparameters.
2. Verify that the correct model implementation and checkpoint are used.
3. Use consistent training/test partitions for baseline and ablation comparisons.
4. Confirm that normalization and material encoders are fitted only on the training partition.
5. Check whether data augmentation is enabled in the selected script.
6. Update hard-coded dataset, checkpoint, and Optuna study paths before execution.
7. Compare results against the paper's reported metrics rather than assuming that a script will automatically reproduce the published table.

In particular, `new_pgnn/test.py` expects an existing Optuna study file at a configured path. Several plotting and inference scripts also refer to historical checkpoint directories. These paths must be updated to match the artifacts generated by the current experiment.

The paper describes six principal physical inputs, whereas the current `new_pgnn/models.py` implementation also contains a material-category embedding. When reproducing the manuscript strictly, verify the intended model version and configuration rather than assuming that all historical scripts implement an identical architecture.

## 13. Reference

Zhang, Y., Miao, Y., Feng, Y., and Cheng, H. *Interpretable subsurface damage prediction for optical components via physics-guided neural network adaptive symbolic layer*. Optical Engineering, 2026.

The final publication metadata, including the DOI, volume, issue, and page numbers, should be added once the official publication record is available.

## 14. License

No `LICENSE` file is currently included in the repository. The permitted use, modification, and redistribution of the code and dataset should therefore be confirmed with the repository owner before redistribution.

