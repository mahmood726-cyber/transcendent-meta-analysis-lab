import pandas as pd
import numpy as np
import scipy.linalg as la
from scipy.spatial.distance import cdist
import os
import sys
import warnings

warnings.filterwarnings("default")


def sinkhorn_knopp(C, reg=0.1, num_iters=100):
    n, m = C.shape
    a, b = np.ones(n)/n, np.ones(m)/m
    K = np.exp(-C / reg)
    u = np.ones(n)/n
    for _ in range(num_iters):
        v = b / (K.T @ u + 1e-9)
        u = a / (K @ v + 1e-9)
    return np.diag(u) @ K @ np.diag(v)


def pool_quantum(y, v):
    """
    Exploratory pooling method using quantum density matrix analogy.
    Maps study effects to 2x2 density matrices, averages in log-space
    (Log-Euclidean mean on the positive-definite manifold), then extracts
    the pooled effect.  NOTE: This is an experimental approach, not a
    standard meta-analytic method (cf. inverse-variance or REML).
    """
    if len(y) == 0:
        return 0.0
    rho_list = []
    for yi, vi in zip(y, v):
        mu = np.tanh(yi)
        # Map variance to purity in (0,1]: high variance -> low purity (mixed state)
        purity = np.clip(1.0 / (1.0 + vi), 0.01, 0.99)
        rho = 0.5 * np.array([[1 + mu*purity, 0], [0, 1 - mu*purity]])
        rho_list.append(rho)
    log_sum = np.zeros((2, 2))
    for r in rho_list:
        log_sum += la.logm(r).real
    rho_mean = la.expm(log_sum / len(rho_list))
    rho_mean /= np.trace(rho_mean)
    val = np.real(np.trace(rho_mean @ np.array([[1, 0], [0, -1]])))
    return np.arctanh(np.clip(val, -0.9999, 0.9999))


def main():
    print("===================================================================")
    print("  COLORECTAL CANCER ONCOLOGY SYNTHESIS (EXPLORATORY)")
    print("===================================================================")

    # 1. Load Data
    DATA_DIR = os.environ.get("TMAL_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
    df = pd.read_csv(os.path.join(DATA_DIR, "all_real_data_integrated.csv"))
    topic_data = df[(df['source'] == 'BMC_2022') & (df['domain'] == 'Colorectal Cancer')].copy()
    topic_data['log_or'] = pd.to_numeric(topic_data['log_or'], errors='coerce')
    topic_data['se'] = pd.to_numeric(topic_data['se'], errors='coerce')
    topic_data = topic_data.dropna(subset=['log_or', 'se'])

    print(f"Total trials in Colorectal Cancer domain: {len(topic_data)}")

    if len(topic_data) < 2:
        print("Insufficient data for synthesis. Exiting.")
        return

    # 2. Precision-based filtering (top 60% most precise studies)
    data_filtered = topic_data.sort_values('se').head(max(1, int(len(topic_data) * 0.6)))
    print(f"Precision-filtered to core studies (N={len(data_filtered)})")

    # 3. Sinkhorn Optimal Transport (RCT vs NRS calibration)
    rct = data_filtered[data_filtered['design'] == 'RCT']
    nrs = data_filtered[data_filtered['design'] != 'RCT']

    if len(rct) > 0 and len(nrs) > 0:
        print(f"Calibrating {len(nrs)} observational studies against {len(rct)} RCTs via optimal transport...")
        C = cdist(rct[['log_or', 'se']].values, nrs[['log_or', 'se']].values, metric='sqeuclidean')
        P = sinkhorn_knopp(C, reg=0.1)
        # Barycentric projection: normalize each NRS point by its coupling mass
        col_sums = P.sum(axis=0)
        col_sums = np.where(col_sums > 1e-12, col_sums, 1e-12)
        synthetic_nrs_y = (P.T @ rct['log_or'].values) / col_sums
        final_y = np.concatenate([rct['log_or'].values, synthetic_nrs_y])
        final_v = np.concatenate([rct['se'].values**2, nrs['se'].values**2])
        print("Optimal Transport calibration complete.")
        print("NOTE: Transported NRS values are approximations; original NRS SEs are retained.")
    else:
        print("Insufficient design diversity. Pooling raw estimates.")
        final_y = data_filtered['log_or'].values
        final_v = data_filtered['se'].values**2

    # 4. Quantum-analogy Pooling
    result = pool_quantum(final_y, final_v)
    print(f"\nPOOLED ONCOLOGY EFFECT (Log OR, exploratory): {result:.6f}")

    # 5. Adversarial Leave-One-Out Influence Check
    max_shift = 0.0
    infl_idx = 0
    result_sens = result  # initialize to pooled result
    for i in range(len(final_y)):
        loo_y = np.delete(final_y, i)
        loo_v = np.delete(final_v, i)
        loo_result = pool_quantum(loo_y, loo_v)
        shift = abs(result - loo_result)
        if shift > max_shift:
            max_shift = shift
            infl_idx = i
            result_sens = loo_result

    print(f"Most Influential Trial: {infl_idx} (LOO Shift: {max_shift:.6f})")
    print(f"Sensitivity Estimate (without trial {infl_idx}): {result_sens:.6f}")

    print("\n===================================================================")
    print("  ONCOLOGY SYNTHESIS COMPLETE")
    print("===================================================================")


if __name__ == '__main__':
    main()
