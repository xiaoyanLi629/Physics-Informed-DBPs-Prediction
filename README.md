# Physics-Informed Explainable Deep Learning for Disinfection By-Products Prediction

A comprehensive machine learning framework for predicting disinfection by-products (DBPs) in drinking water treatment, featuring a physics-informed explainable deep learning model with attention-based feature group importance analysis.

## Highlights

- **14 models** compared: classical ML, deep learning, KAN, TabNet, and an explainable model
- **Physics-informed architecture**: feature grouping based on DBPs formation chemistry
- **Attention mechanism**: learns interpretable feature group importance weights
- **Ablation study**: 9 configurations validating each model component
- **Rigorous evaluation**: 5-fold cross-validation, Wilcoxon signed-rank tests

## Project Structure

```
DBPs/
├── src/
│   ├── models.py                  # All model definitions and evaluator
│   ├── run.py                     # Main entry point
│   └── generate_paper_figures.py  # Publication-quality figure generation
├── data/
│   └── data.csv                   # Dataset (66 samples, 8 features, 5 targets)
├── results/
│   ├── figures/                   # Model prediction plots and training curves
│   └── summary/                   # CSV results, reports, and logs
├── manuscript/                    # LaTeX source and compiled PDF
│   ├── Physics_Informed_Explainable_DL_for_DBPs_Prediction.tex
│   ├── Physics_Informed_Explainable_DL_for_DBPs_Prediction.pdf
│   ├── references.bib
│   └── figure/                    # Figures used in the paper
├── statistics/                    # Exploratory data analysis scripts and outputs
├── references/                    # Reference papers
├── requirements.txt
└── .gitignore
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Full Evaluation

```bash
python src/run.py
```

This will train all 14 models, run ablation studies, perform 5-fold cross-validation, and generate results in the `results/` directory.

### Generate Paper Figures

```bash
python src/generate_paper_figures.py
```

## Features and Targets

**Input features** (8):
Temp, pH, UVA254, Cl2, NO2-N, DOC, NH4-N, Br

**Target variables** (5 HAAs):
DCAA, TCAA, BCAA, HAA5, HAA9

## Explainable Model Architecture

The model groups input features by their physicochemical role in DBPs formation:

| Group | Features | Chemical Role |
|-------|----------|---------------|
| Organic Matter | DOC, UVA254 | Primary organic precursors |
| Nitrogen Compounds | NO2-N, NH4-N | Competitive chlorine consumers |
| Environmental | Temp, pH | Reaction kinetics control |
| Halides | Br | Brominated DBPs pathway |
| Disinfectant | Cl2 | Oxidant driving force |

An attention mechanism learns the importance of each group, providing interpretable predictions aligned with domain knowledge.

## Key Results

### Model Performance (Test Set)

| Model | Test R² | Test MAE | Note |
|-------|---------|----------|------|
| SVR | 0.4013 | 0.4538 | Best generalization |
| Explainable Model | 0.3034 | 0.4080 | Best MAE, interpretable |
| TabNet | 0.3235 | 0.4867 | Modern baseline |

### Learned Attention Weights

| Feature Group | Weight | Interpretation |
|---------------|--------|----------------|
| Organic Matter | 44.0% | Dominant factor, consistent with chemistry |
| Nitrogen Compounds | 32.6% | Competitive reactions with free chlorine |
| Environmental | 14.7% | Kinetics regulation (Arrhenius, pH equilibrium) |
| Halides | 5.9% | Brominated species selectivity |
| Disinfectant | 2.8% | Narrow operational range in practice |

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{dbps2026,
  title={Physics-Informed Explainable Deep Learning for Disinfection By-Products Prediction},
  booktitle={Lecture Notes in Computer Science},
  year={2026}
}
```

## License

This project is for academic research purposes.
