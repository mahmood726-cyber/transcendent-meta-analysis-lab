import numpy as np
from scipy.special import logsumexp

from core.data_loader import load_integrated_df


def compute_path_signature(y, order=2):
    """
    Rough Path Signature (cf. Hairer, Lyons).
    Computes the level-1 and level-2 iterated integrals of the
    chronologically ordered evidence path.
    """
    if len(y) < 2:
        return np.array([1.0, 0.0, 0.0])

    dy = np.diff(y)
    S1 = np.sum(dy)

    # Level 2: iterated integral (area under cumulative-sum curve)
    S2 = 0.0
    for i in range(1, len(dy)):
        S2 += np.sum(dy[:i]) * dy[i]

    return np.array([1.0, S1, S2])


def perfectoid_tilting(rct_y, nrs_y):
    """
    Exploratory analogy with Perfectoid Spaces (Scholze).
    Applies iterated Frobenius-like compression to NRS data, then a single
    inverse step.  The residual gap measures how much the NRS distribution
    differs from RCTs after this lossy transformation.
    NOTE: This is a mathematical analogy, not a bias-adjustment method.
    Standard approaches include IPTW, propensity scores, or bias models.
    """
    if len(rct_y) == 0 or len(nrs_y) == 0:
        return np.nan

    p = 3
    depth = 5

    tilted = np.copy(nrs_y).astype(float)
    for _ in range(depth):
        tilted = np.sign(tilted) * (np.abs(tilted) ** (1.0 / p))

    untilted = np.sign(tilted) * (np.abs(tilted) ** p)

    return np.mean(rct_y) - np.mean(untilted)


def anabelian_iutt_deformation(y):
    """
    Measures the gap between the arithmetic mean of log-ORs and the
    log of the arithmetic mean of ORs (Jensen's inequality gap).
    Larger gaps indicate greater heterogeneity in the effect distribution.
    Named by analogy with IUTT additive-multiplicative deformation.
    """
    additive_mean = np.mean(y)

    # Numerically stable log-of-mean-exp via logsumexp
    multiplicative_mean = logsumexp(y) - np.log(len(y))

    theta_link_discrepancy = np.abs(additive_mean - multiplicative_mean)

    return theta_link_discrepancy


def analyze_domain(name, data):
    print(f"\n--- ANALYZING DOMAIN: {name} ---")
    data = data.sort_values('year')
    y = data['log_or'].values
    v = data['se'].values**2

    print(f"Total Trials: {len(y)}")

    # 1. Rough Path Signature
    sig = compute_path_signature(y)
    print("1. Path Signature (Rough Path Theory analogy):")
    print(f"   Level 1 (Total Drift): {sig[1]:.4f}")
    print(f"   Level 2 (Cumulative Volatility): {sig[2]:.4f}")
    if abs(sig[2]) > abs(sig[1]):
        print("   OBSERVATION: Evidence path shows high volatility relative to net drift.")
    else:
        print("   OBSERVATION: Evidence path has a stable directional drift.")

    # 2. Perfectoid Tilting
    rct_y = data[data['design'] == 'RCT']['log_or'].values
    nrs_y = data[data['design'] != 'RCT']['log_or'].values
    tilt_gap = perfectoid_tilting(rct_y, nrs_y)
    print("\n2. RCT-NRS Gap (Perfectoid analogy):")
    if np.isnan(tilt_gap):
        print("   Insufficient mixed-design data for comparison.")
    else:
        print(f"   Residual gap after Frobenius transformation: {tilt_gap:.6f}")
        if abs(tilt_gap) < 0.1:
            print("   OBSERVATION: Small gap suggests RCT and NRS estimates are consistent.")
        else:
            print("   OBSERVATION: Large gap suggests meaningful RCT-NRS discrepancy.")

    # 3. Jensen's Inequality Gap (IUTT analogy)
    theta_def = anabelian_iutt_deformation(y)
    print("\n3. Additive-Multiplicative Deformation (Jensen gap):")
    print(f"   |mean(log OR) - log(mean(OR))|: {theta_def:.6f}")
    if theta_def > 0.05:
        print("   OBSERVATION: Large Jensen gap indicates substantial effect heterogeneity.")
        print("   Scale-dependent aggregation may yield different conclusions.")
    else:
        print("   OBSERVATION: Additive and multiplicative scales are approximately consistent.")


def main():
    print("===================================================================")
    print("  EXPLORATORY ANALYSIS: PATH SIGNATURES, RCT-NRS GAP & HETEROGENEITY")
    print("===================================================================")

    cohorts = [
        ("Breast Cancer (BMC_2022)", {"source": "BMC_2022", "domain": "Breast Cancer"}),
        ("Massive Clinical Network: Domain 7 (IMPACT_HTA)",
         {"source": "IMPACT_HTA", "domain": "7"}),
    ]
    for label, filt in cohorts:
        try:
            cohort = load_integrated_df(**filt)
        except (FileNotFoundError, ValueError) as e:
            print(f"No data found for {label}: {e}")
            continue
        print(f"Loaded {label} cohort: {len(cohort)} trials.")
        analyze_domain(label, cohort)

    print("\n===================================================================")
    print("  EXPLORATORY ANALYSIS COMPLETE.")
    print("===================================================================")


if __name__ == '__main__':
    main()
