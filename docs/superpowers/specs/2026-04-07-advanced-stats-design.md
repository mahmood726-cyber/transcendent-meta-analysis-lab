# Advanced Statistics & Mathematics Module — Design Spec

**Date**: 2026-04-07
**Project**: Transcendent-Meta-Analysis-Lab
**Scope**: 10 new methods (5 validated + 5 exotic), layered architecture, ~130 tests
**Data**: Existing CSVs (`all_real_data_integrated.csv`, `ma4_results_pairwise70.csv`)

---

## 1. Architecture

```
methods/                       # Validated statistical layer (peer-review grade)
  __init__.py
  copula_ma.py                 # 1. Copula multivariate MA
  dirichlet_process.py         # 2. Dirichlet Process mixture clustering
  bayesian_model_avg.py        # 3. Bayesian Model Averaging
  robust_variance.py           # 4. Hedges-Tipton sandwich estimator (CR2)
  cumulative_changepoint.py    # 5. Cumulative MA + change-point detection

exploration/                   # Exotic mathematical layer (exploratory)
  __init__.py
  wasserstein_barycenter.py    # 6. Wasserstein-2 barycenters
  persistent_homology.py       # 7. TDA / Betti numbers
  fisher_rao.py                # 8. Information geometry geodesics
  tropical_pooling.py          # 9. Min-plus tropical consensus
  spectral_graph.py            # 10. Laplacian eigenmaps

core/                          # Shared infrastructure
  __init__.py
  types.py                     # MAData / MAResult dataclasses
  data_loader.py               # CSV loading, validation, filtering
  validation.py                # Numeric guards, edge case helpers

tests/
  conftest.py                  # Shared fixtures (real + synthetic)
  test_copula_ma.py
  test_dirichlet_process.py
  test_bayesian_model_avg.py
  test_robust_variance.py
  test_cumulative_changepoint.py
  test_wasserstein_barycenter.py
  test_persistent_homology.py
  test_fisher_rao.py
  test_tropical_pooling.py
  test_spectral_graph.py
  test_core.py                 # data_loader, types, validation
```

### Design Principles

- Every method takes `MAData` and returns `MAResult`
- No external ML/TDA libraries — pure numpy/scipy implementations
- Fixed seeds everywhere (`np.random.seed` or `rng` parameter)
- All iterative methods accept `max_iter`, `tol`, `seed` parameters
- `exploration/` modules clearly labeled as non-standard in docstrings
- Each module is self-contained (no cross-imports between methods)

---

## 2. Core Infrastructure

### `core/types.py`

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MAData:
    """Standard input for all meta-analysis methods."""
    effect: np.ndarray          # effect sizes (log-OR, SMD, etc.)
    se: np.ndarray              # standard errors
    group: np.ndarray | None = None   # review_id or grouping var
    year: np.ndarray | None = None    # publication year (for temporal methods)
    design: np.ndarray | None = None  # 'RCT' / 'NRS' labels

    def __post_init__(self):
        assert len(self.effect) == len(self.se), "effect and se must have same length"
        assert len(self.effect) > 0, "MAData must have at least 1 study"
        assert np.all(self.se > 0), "All standard errors must be positive"

@dataclass
class MAResult:
    """Standard output from all meta-analysis methods."""
    estimate: float             # pooled effect
    ci_lower: float             # lower confidence/credible interval
    ci_upper: float             # upper confidence/credible interval
    method: str                 # method name for identification
    details: dict = field(default_factory=dict)  # method-specific extras
```

### `core/data_loader.py`

- `load_integrated(domain=None, source=None, design=None) -> MAData`
  - Reads `all_real_data_integrated.csv` from `TMAL_DATA_DIR` or `./data/`
  - Columns: `log_or` -> effect, `se` -> se, `source`+`domain` -> group, `year`, `design`
  - Validates column existence, coerces numeric, drops NaN on effect+se
  - Raises `ValueError` if result has 0 studies
  - Warns if < 3 studies

- `load_pairwise70(sample_n=None, seed=42) -> MAData`
  - Reads `ma4_results_pairwise70.csv`
  - Columns: `theta` -> effect, `sigma` -> se, `review_id` -> group
  - Optional: sample `sample_n` reviews for performance

### `core/validation.py`

- `require_min_studies(data: MAData, k: int)` — raises if fewer than k studies
- `safe_exp(x)` — clamps to avoid overflow (max 700)
- `safe_log(x)` — clamps to avoid log(0) (min 1e-300)
- `logsumexp_mean(x)` — numerically stable log(mean(exp(x)))
- `ensure_positive_definite(M)` — nearest PD matrix via Higham (2002) algorithm
- `check_convergence(values, tol)` — True if last N values differ by < tol

---

## 3. Validated Methods Layer

### Method 1: Copula Multivariate MA (`copula_ma.py`)

**Purpose**: Model correlated outcomes within the same review using a Gaussian copula.

**Algorithm**:
1. For each review with multiple outcomes, estimate pairwise correlations from effect-SE overlap
2. Build a block-diagonal working correlation matrix (one block per review)
3. GLS estimation: β = (X'V⁻¹X)⁻¹ X'V⁻¹y where V = block-diag(Σ_review + τ²I)
4. Profile likelihood for τ² via REML
5. When within-review correlation is unknown, use EM to impute from marginal structure

**Input**: MAData with `group` set (reviews with multiple outcomes)
**Output**: MAResult with details = `{tau2, correlation_matrix, per_review_estimates, n_reviews, n_outcomes}`

**Edge cases**:
- Reviews with single outcome: treated as independent (correlation=0)
- Singular correlation matrix: regularize with `ensure_positive_definite`
- All reviews have 1 outcome: falls back to standard univariate RE

**Validation targets**: `metafor::rma.mv` with `V = vcalc()` on Cochrane data

---

### Method 2: Dirichlet Process Mixture (`dirichlet_process.py`)

**Purpose**: Discover latent subgroups in the effect distribution without pre-specifying the number of clusters.

**Algorithm** (Gibbs sampler, stick-breaking):
1. Stick-breaking prior: weights w_k = v_k * prod(1 - v_j, j<k), v_k ~ Beta(1, alpha)
2. Component parameters: μ_k ~ N(0, 1), σ_k² ~ InvGamma(2, 1)
3. Assignments: z_i ~ Categorical(w) based on likelihood N(y_i | μ_{z_i}, σ_{z_i}² + se_i²)
4. Truncation at K=20 components
5. Gibbs sweep: update z, then μ_k, then σ_k², then v_k, then alpha
6. Run 1000 iterations, 500 burn-in, thin by 2

**Input**: MAData (group not required)
**Output**: MAResult (estimate = posterior mean of mixture) with details = `{cluster_assignments, cluster_means, cluster_variances, cluster_weights, alpha, n_clusters_active, trace}`

**Edge cases**:
- k=1: single cluster, returns that study's effect
- All identical effects: single cluster, tau²=0
- Extreme outlier: gets its own singleton cluster

**Validation targets**:
- Bimodal simulation: N(-1, 0.1²) x 10 + N(0.5, 0.1²) x 10 → should discover 2 clusters
- Unimodal simulation: N(0, 0.5²) x 20 → should return 1-2 clusters (not fragment)

---

### Method 3: Bayesian Model Averaging (`bayesian_model_avg.py`)

**Purpose**: Account for model uncertainty by averaging across multiple pooling methods.

**Algorithm**:
1. Fit 4 models, each returning (estimate, tau², log-likelihood):
   - Fixed-effect (IV-weighted, tau²=0)
   - DerSimonian-Laird (method of moments)
   - REML (restricted maximum likelihood via Newton-Raphson)
   - Paule-Mandel (iterative generalized Q)
2. Compute BIC for each: BIC = -2*loglik + p*log(k) where p=2 (μ, τ²), k=n_studies
3. Model weights: w_m = exp(-BIC_m/2) / sum(exp(-BIC_j/2))
4. BMA estimate = sum(w_m * estimate_m)
5. BMA variance = sum(w_m * (var_m + (estimate_m - BMA_est)²))

**Input**: MAData (group not required)
**Output**: MAResult with details = `{models: [{name, estimate, tau2, bic, weight}], bma_variance}`

**Edge cases**:
- k=1: only fixed-effect is identifiable, weight=1.0
- tau²=0 in all models: FE dominates
- REML fails to converge: fall back to DL, set REML weight to 0

**Validation targets**: Manual BIC calculation on BCG vaccine dataset (13 studies, well-known)

---

### Method 4: Robust Variance Estimation (`robust_variance.py`)

**Purpose**: Hedges-Tipton-Pustejovsky CR2 sandwich estimator for dependent effects.

**Algorithm**:
1. Fit working model: β = (X'WX)⁻¹ X'Wy where W = diag(1/vi)
2. Residuals: e = y - Xβ
3. Meat matrix: M = sum over clusters j of (A_j * e_j * e_j' * A_j')
   where A_j = (I - H_j)⁻¹/² is the CR2 bias-correction multiplier
   and H_j is the leverage submatrix for cluster j
4. Sandwich: V_robust = (X'WX)⁻¹ M (X'WX)⁻¹
5. Satterthwaite df: ν = 2*E[V_robust]² / Var[V_robust] (approximated)

**Input**: MAData with `group` set (clusters of dependent effects)
**Output**: MAResult with details = `{robust_se, df_satterthwaite, naive_se, n_clusters, leverage}`

**Edge cases**:
- Single cluster: degenerates to naive SE (df=0, warn user)
- < 4 clusters: Satterthwaite df may be < 1, clamp to df=1
- Independent effects (all singletons): equals naive IV-weighted SE

**Validation targets**: `clubSandwich::coef_test(model, vcov="CR2")` on Cochrane data

---

### Method 5: Cumulative MA + Change-Point (`cumulative_changepoint.py`)

**Purpose**: Track how the pooled estimate evolves chronologically and detect structural breaks.

**Algorithm**:
1. Sort studies by `year` (ties broken by SE ascending)
2. Cumulative: at step t, pool studies 1..t via IV-weighted FE. Record estimate_t, se_t, z_t.
3. CUSUM: S_t = sum(z_i - z_bar, i=1..t). Detect drift when |S_t| > threshold (h = 4*σ).
4. Bayesian online change-point detection (Adams & MacKay 2007):
   - Hazard function: H(τ) = 1/λ (geometric prior on run length, λ = mean run length)
   - Run-length posterior: P(r_t | y_{1:t}) updated recursively
   - Change-point at t if P(r_t=0 | y_{1:t}) > 0.5
5. Returns both CUSUM and Bayesian detections

**Input**: MAData with `year` set
**Output**: MAResult (estimate = final cumulative) with details = `{cumulative_effects, cumulative_ses, cusum_values, cusum_alarms, bayesian_changepoints, run_length_posterior}`

**Edge cases**:
- No year data: raise ValueError("Cumulative MA requires year data")
- All same year: returns single pooled estimate, no change-points possible
- Monotonically stable evidence: no alarms, empty change-point list

**Validation targets**: Simulated data with known shift at index 15 of 30 studies (effect jumps from 0 to -0.5)

---

## 4. Exotic Mathematical Layer

### Method 6: Wasserstein Barycenters (`wasserstein_barycenter.py`)

**Purpose**: Compute the optimal transport center of study-level effect distributions.

**Algorithm** (closed-form for Gaussians):
1. Each study i defines a distribution N(y_i, v_i) where v_i = se_i²
2. W2 distance between Gaussians: W2²(P,Q) = (μ_P - μ_Q)² + (σ_P - σ_Q)²
3. Barycenter mean: μ_bar = sum(w_i * y_i) (same as IV-weighted for equal weights)
4. Barycenter variance via fixed-point iteration:
   σ_bar = sum(w_i * sqrt(σ_bar * v_i)) — iterate until convergence
5. Weights: w_i = (1/v_i) / sum(1/v_j) (precision-weighted)
6. Per-study transport cost: W2²(study_i, barycenter)
7. Outlier flag: cost > 3 * median(costs)

**Input**: MAData
**Output**: MAResult with details = `{barycenter_var, transport_costs, outlier_flags, n_iterations}`

**Edge cases**:
- k=1: barycenter = that study's distribution
- All v_i equal: barycenter variance = v_i (trivial fixed point)
- v_i = 0: degenerate point mass, clamp to v_min = 1e-10

**Validation targets**:
- 2 Gaussians N(0,1) and N(0,4): barycenter variance should be ((1+2)/2)² = 2.25 (known)
- Equal-weight equal-variance: barycenter mean = arithmetic mean, var = common var

---

### Method 7: Persistent Homology (`persistent_homology.py`)

**Purpose**: Topological data analysis of the study-similarity complex to detect clusters and contradictions.

**Algorithm**:
1. Embed studies as points in (effect/max_se, se/max_se) space (normalized)
2. Build Vietoris-Rips filtration: at radius ε, connect studies with distance < ε
3. Track connected components (β0) and loops (β1) across ε from 0 to max_distance
4. Persistence: feature born at ε_birth, dies at ε_death. Persistence = death - birth.
5. Implementation: incremental union-find for β0, boundary matrix reduction for β1
6. No external TDA library — pure numpy for 2D point clouds

**Output dataclass**:
```python
@dataclass
class TDAResult:
    persistence_pairs_0: list  # (birth, death) for H0 (components)
    persistence_pairs_1: list  # (birth, death) for H1 (loops)
    betti_curve_0: np.ndarray  # β0(ε) across filtration
    betti_curve_1: np.ndarray  # β1(ε) across filtration
    filtration_values: np.ndarray  # ε values
```

**Input**: MAData
**Output**: MAResult with details containing TDAResult fields

**Edge cases**:
- k < 3: no loops possible (β1=0 always)
- All studies identical: single component at ε=0
- > 500 studies: subsample to 500 (Rips complex is O(n³) in memory)

**Validation targets**:
- 3 clusters of 5 studies each (well-separated): β0 should show 3 persistent components
- Triangle configuration (3 equidistant studies): β1=1 loop at the right ε
- Line configuration: β0 drops from n to 1, β1=0 always

---

### Method 8: Fisher-Rao Geodesics (`fisher_rao.py`)

**Purpose**: Exact information-geometric distances and means on the normal distribution manifold.

**Algorithm**:
1. Fisher-Rao distance between N(μ1,σ1²) and N(μ2,σ2²):
   Use the Skovgaard (1984) / Costa-Santos-Strapasson (2015) formula:
   - d = √2 * arccosh( ((μ1-μ2)² + 2(σ1² + σ2²)) / (4*σ1*σ2) )
   - Exact when σ1=σ2, tight approximation otherwise (error < 1% for σ1/σ2 < 3)
   - Argument to arccosh is always ≥ 1 by AM-GM inequality, so d ≥ 0
   - Special cases: equal σ → d ∝ |μ1-μ2|/σ; equal μ → d = √2*|log(σ1/σ2)|

2. Frechet mean: minimize sum(w_i * d²(θ, θ_i)) via Riemannian gradient descent
   - Gradient in (μ,σ) coordinates derived from the Fisher metric tensor
   - Step size via Armijo backtracking line search
   - Converges for convex combinations on the half-plane

3. Geodesic interpolation: γ(t) between two normals on the Fisher manifold

**Input**: MAData
**Output**: MAResult (estimate = Frechet mean μ) with details = `{frechet_sigma, distance_matrix, geodesic_paths, n_gradient_steps}`

**Edge cases**:
- All same σ: reduces to weighted mean of μ (Euclidean in μ direction)
- All same μ: reduces to geometric-like mean of σ
- σ_i very small: clamp to σ_min = 1e-6

**Validation targets**:
- Equal variances: Frechet mean μ should equal IV-weighted mean
- Equal means: Frechet σ should be close to geometric mean of σ_i
- Two distributions: geodesic midpoint should be computable analytically

---

### Method 9: Tropical Geometry (`tropical_pooling.py`)

**Purpose**: Outlier-immune consensus via min-plus algebra.

**Algorithm**:
1. Tropical semiring: (R ∪ {∞}, min, +) — "addition" = min, "multiplication" = +
2. Weighted effects: w_i = -log(v_i) (precision in tropical space)
3. Tropical weighted mean: minimize max_i(|θ - y_i| + w_i) — the Chebyshev center
   Solved via linear programming: min t s.t. |θ - y_i| + w_i ≤ t for all i
   Equivalent to: min t s.t. y_i - w_i - t ≤ θ ≤ y_i + w_i + t for all i
   → θ = (max(y_i - w_i) + min(y_i + w_i)) / 2, t = (min(y_i+w_i) - max(y_i-w_i)) / 2
4. Tropical convex hull: vertices are studies that are tropical-extreme
   A study is tropical-extreme if removing it changes the tropical polytope
5. Robustness score: fraction of studies that are NOT tropical-extreme (interior)

**Input**: MAData
**Output**: MAResult with details = `{tropical_median, tropical_radius, hull_vertices, hull_interior, robustness_score}`

**Edge cases**:
- k=1: tropical median = that study, radius = 0
- All equal weights: tropical median = standard minimax (midrange)
- Single outlier: only affects result if it's the most extreme — inherent robustness

**Validation targets**:
- 3 studies at (-1, 0, 1) with equal SE: tropical median = 0, radius = 1
- Outlier test: 10 studies near 0 + 1 study at 10 → tropical median barely affected
- Compare robustness to standard IV-weighted (tropical should resist outliers better)

---

### Method 10: Spectral Graph Methods (`spectral_graph.py`)

**Purpose**: Study-network analysis via graph Laplacian eigenmaps.

**Algorithm**:
1. Similarity graph: W_ij = exp(-d_ij² / (2h²)) where
   d_ij = |y_i - y_j| / sqrt(v_i + v_j) (standardized effect distance)
   h = median(d_ij) (bandwidth)
2. Degree matrix: D_ii = sum_j W_ij
3. Normalized Laplacian: L_norm = I - D^(-1/2) W D^(-1/2)
4. Eigendecomposition: L_norm = U Λ U'
5. Key outputs:
   - Fiedler value (λ2): evidence connectivity (0 = disconnected, ~1 = well-connected)
   - Spectral gap (λ2): larger gap = clearer consensus
   - k clusters via k-means on first k eigenvectors (k chosen by eigengap heuristic)
   - 2D embedding from first 2 non-trivial eigenvectors
6. Graph-weighted pooled estimate: weight each study by its degree centrality

**Input**: MAData
**Output**: MAResult (estimate = degree-centrality-weighted mean) with details = `{fiedler_value, spectral_gap, cluster_labels, n_clusters, embedding_2d, degree_centrality}`

**Edge cases**:
- k < 3: trivial graph, return simple mean
- All identical effects: W = all ones, λ2 = 1 (perfect consensus)
- Disconnected components: Fiedler value = 0, warn user

**Validation targets**:
- Two well-separated groups: Fiedler ≈ 0, 2 clusters detected
- Homogeneous set: Fiedler near 1, 1 cluster
- Star configuration (1 hub, many spokes): hub has highest centrality

---

## 5. Testing Strategy

### Fixture Design (`tests/conftest.py`)

| Fixture | k | Description | Used by |
|---------|---|-------------|---------|
| `small_5` | 5 | Hand-calculated, exact reference values | All methods |
| `bimodal_20` | 20 | Two N(-1,0.1²) + N(0.5,0.1²) groups | DP, Spectral, TDA, Wasserstein |
| `homogeneous_10` | 10 | N(0,0.3²), tau²=0 | BMA, Robust, Copula |
| `single_study` | 1 | k=1 edge case | All methods |
| `pair_studies` | 2 | k=2 edge case | All methods |
| `temporal_30` | 30 | Known shift at index 15 | Cumulative, Change-point |
| `with_outlier` | 11 | 10 near 0 + 1 at 5.0 | Tropical, Wasserstein, Spectral |
| `real_oncology` | ~50 | Colorectal Cancer from CSV | Integration tests |
| `real_domain7` | ~200 | IMPACT_HTA Domain 7 from CSV | Integration tests |
| `clustered_reviews` | 70 | 10 reviews x 7 outcomes | Copula, Robust |

### Test Categories per Method (~12-15 each)

1. **Reference correctness** (3): Output matches R/analytic/hand-calculated values within tol=1e-4
2. **Edge cases** (3): k=1, k=2, tau²=0, empty group, extreme SE (>100)
3. **Convergence** (2): Iterative methods converge within max_iter; result stable across tol values
4. **Properties** (2): CI contains estimate; distances non-negative; weights sum to 1; Betti numbers non-negative
5. **Consistency** (2): Stable across seeds (within tolerance); subsample gives similar direction
6. **Integration** (1-2): Runs on real data fixture without error; output is finite

### Target: ~130 tests, all passing

---

## 6. Dependencies

No new external dependencies. Everything built on existing stack:
- `numpy>=1.23,<2.0`
- `scipy>=1.9,<2.0` (for linalg, optimize, special, stats)
- `pandas>=1.5,<3.0` (for data loading only)

---

## 7. Non-Goals

- No visualization (plotting is out of scope)
- No HTML dashboard for these methods (separate project if needed)
- No R implementations (Python only)
- No GPU acceleration
- No streaming/online versions (batch only)
- No integration with existing root-level scripts (those remain independent)
