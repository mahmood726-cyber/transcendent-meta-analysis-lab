# Transcendent Meta-Analysis Lab

Exploratory project applying advanced mathematical frameworks (information theory, optimal transport, path signatures, Bayesian particle inference) as analogies for understanding patterns in clinical evidence synthesis.

**Important**: The mathematical frameworks used here (quantum density matrices, perfectoid tilting, Chaitin's constant analogies) are exploratory metaphors applied to clinical trial data. They are not established statistical methods for evidence synthesis and have not been validated for clinical decision-making. Standard meta-analytic methods (inverse-variance, REML, DerSimonian-Laird) should be used for actual clinical conclusions. This project does not follow PRISMA reporting guidelines.

## Methods Explored

### Standard Statistical Methods (R)
- **Bayesian Hierarchical Meta-Analysis** (brms, Student-t likelihood)
- **Multi-Level Random Effects** (metafor `rma.mv`)
- **Meta-Regression** (moderator analysis)
- **Shannon Entropy** of effect directionality

### Exploratory Mathematical Analogies (Python)
- **Density Matrix Pooling**: Log-Euclidean mean on 2x2 positive-definite manifold
- **Optimal Transport Calibration**: Sinkhorn-Knopp for RCT-NRS alignment
- **Path Signatures**: Rough-path iterated integrals of chronological evidence
- **Frobenius Compression**: Iterated p-th root transformation (perfectoid analogy)
- **Jensen Gap**: Additive vs. multiplicative scale deformation (IUTT analogy)
- **Lempel-Ziv Complexity**: Algorithmic compressibility of effect-sign sequences
- **SVGD Particle Inference**: Non-parametric posterior estimation

## Data

Analyzed ~2,900 clinical trials across Oncology, Pain Management, and HTA domains.

## Repository Structure

- `data/` - Place CSV data files here (or set `TMAL_DATA_DIR` / `TMAL_R_DATA_DIR` env vars)
- `the_absolute_limit.py` - Algorithmic complexity analysis (Lempel-Ziv, binary encoding)
- `omniscience_tier_synthesis.py` - SVGD particle inference + structural regularity tests
- `epistemic_horizon_synthesis.py` - Path signatures, RCT-NRS gap, heterogeneity measures
- `colorectal_cancer_synthesis.py` - Optimal transport calibration + density matrix pooling
- `massive_r_synthesis.R` / `_v2.R` / `_v3.R` - Bayesian hierarchical models (R/brms/metafor)
- `miracle_deep_dive.R` - Opiate reduction time-series analysis

## Setup

```bash
pip install -r requirements.txt
# Place data in ./data/ or set environment variables:
export TMAL_DATA_DIR=/path/to/python/data
export TMAL_R_DATA_DIR=/path/to/r/data
```

R scripts require: `metafor`, `brms`, `Matrix`

---
**Status**: Exploratory synthesis complete.
