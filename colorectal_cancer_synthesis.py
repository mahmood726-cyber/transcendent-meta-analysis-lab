import pandas as pd
import numpy as np
import scipy.linalg as la
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings("ignore")

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
    rho_list = []
    for yi, vi in zip(y, v):
        mu = np.tanh(yi)
        purity = np.clip(1.0 - vi, 0.01, 0.99)
        rho = 0.5 * np.array([[1 + mu*purity, 0], [0, 1 - mu*purity]])
        rho_list.append(rho)
    log_sum = np.zeros((2, 2))
    for r in rho_list:
        log_sum += la.logm(r).real
    rho_mean = la.expm(log_sum / len(rho_list))
    rho_mean /= np.trace(rho_mean)
    val = np.real(np.trace(rho_mean @ np.array([[1, 0], [0, -1]])))
    return np.arctanh(val) if abs(val) < 1 else np.sign(val) * 2.0

print("===================================================================")
print("  NEW TOPIC: COLORECTAL CANCER ONCOLOGY SYNTHESIS")
print("===================================================================")

# 1. Load Data
df = pd.read_csv(r"C:\Users\user\all_real_data_integrated.csv")
topic_data = df[(df['source'] == 'BMC_2022') & (df['domain'] == 'Colorectal Cancer')].copy()
topic_data['log_or'] = pd.to_numeric(topic_data['log_or'], errors='coerce')
topic_data['se'] = pd.to_numeric(topic_data['se'], errors='coerce')
topic_data = topic_data.dropna(subset=['log_or', 'se'])

print(f"Total trials in Colorectal Cancer domain: {len(topic_data)}")

# 2. Renormalization Group (RG) Flow Filter
# Filter to the top 60% precise trials
data_filtered = topic_data.sort_values('se').head(int(len(topic_data) * 0.6))
print(f"RG-Flow coarse-grained to the stable core (N={len(data_filtered)})")

# 3. Sinkhorn Debiasing (RCT vs NRS)
rct = data_filtered[data_filtered['design'] == 'RCT']
nrs = data_filtered[data_filtered['design'] != 'RCT']

if len(rct) > 0 and len(nrs) > 0:
    print(f"Debiasing {len(nrs)} observational studies against {len(rct)} gold-standard RCTs...")
    C = cdist(rct[['log_or', 'se']].values, nrs[['log_or', 'se']].values, metric='sqeuclidean')
    P = sinkhorn_knopp(C, reg=0.1)
    synthetic_nrs_y = (P.T @ rct['log_or'].values) * (len(nrs) / len(rct))
    final_y = np.concatenate([rct['log_or'].values, synthetic_nrs_y])
    final_v = np.concatenate([rct['se'].values**2, nrs['se'].values**2])
    print("Optimal Transport complete.")
else:
    print("Insufficient design diversity. Pooling raw estimates.")
    final_y = data_filtered['log_or'].values
    final_v = data_filtered['se'].values**2

# 4. Quantum Pooling
result = pool_quantum(final_y, final_v)
print(f"\nPOOLED ONCOLOGY EFFECT (Log OR): {result:.6f}")

# 5. Ad-Hoc Adversarial Check
# Check if removing the single most influential trial flips the result
infl_idx = np.argmax(1/final_v)
sensitivity_y = np.delete(final_y, infl_idx)
sensitivity_v = np.delete(final_v, infl_idx)
result_sens = pool_quantum(sensitivity_y, sensitivity_v)
print(f"Sensitivity (Influence of Trial {infl_idx}): {result_sens:.6f} (Shift: {abs(result-result_sens):.6f})")

print("\n===================================================================")
print("  ONCOLOGY SYNTHESIS COMPLETE")
print("===================================================================")
