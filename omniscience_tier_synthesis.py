import numpy as np

from core.data_loader import load_integrated_df

np.random.seed(42)


def _first_n_primes(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return np.array(primes, dtype=float)


def stein_variational_gradient_descent(y, v, n_particles=50, iters=100):
    """
    Bayesian Inference via SVGD (Liu & Wang, NIPS 2016).
    Represents the posterior distribution of the clinical effect as a
    set of particles that repel each other to cover the posterior
    while being pulled toward high-likelihood regions.
    """
    particles = np.random.normal(np.mean(y), 0.5, (n_particles, 1))

    def log_prob_grad(theta):
        # Gradient of log posterior: Normal likelihood + N(0,1) prior
        prior_grad = -theta
        likelihood_grad = np.mean(-(theta - y) / (v + 1e-8))
        return prior_grad + likelihood_grad

    step_size = 0.01
    for _ in range(iters):
        # Squared distance matrix
        norms = np.sum(particles**2, axis=1, keepdims=True)
        sq_dist = norms + norms.T - 2 * (particles @ particles.T)

        # Bandwidth via median heuristic
        bandwidth = np.median(sq_dist) / np.log(n_particles + 1)
        bandwidth = max(bandwidth, 1e-8)  # guard against zero
        K = np.exp(-sq_dist / bandwidth)

        # Kernel gradient
        grad_K = -2 / bandwidth * K @ particles + 2 / bandwidth * particles * np.sum(K, axis=1)[:, np.newaxis]

        # SVGD update
        grads = np.array([log_prob_grad(p) for p in particles])
        phi = (K @ grads + grad_K) / n_particles

        particles += step_size * phi

    return particles


def chow_motive_of_evidence(y):
    """
    Exploratory: maps effect size distribution to L-function-like
    coefficients and checks the Ramanujan-Petersson bound as a
    structural regularity test.  This is a mathematical analogy,
    not a literal application of the Modularity Theorem.
    """
    hist, _ = np.histogram(y, bins=10)
    a_p = hist - np.mean(hist)

    p_vals = _first_n_primes(len(a_p))
    bounds = 2 * np.sqrt(p_vals)
    violations = np.sum(np.abs(a_p) > bounds)

    return violations


def main():
    print("===================================================================")
    print("  EXPLORATORY TIER: STEIN VARIATIONAL INFERENCE & STRUCTURAL TESTS")
    print("===================================================================")

    # Analyze Domain 7
    try:
        data = load_integrated_df(source="IMPACT_HTA", domain="7")
    except (FileNotFoundError, ValueError) as e:
        print(f"Could not load data: {e}")
        return
    y = data['log_or'].values
    v = data['se'].values**2

    print(f"Ingesting {len(y)} trials from Domain 7 for particle-based inference.")

    print("\n1. Stein Variational Gradient Descent (SVGD):")
    particles = stein_variational_gradient_descent(y, v)
    posterior_mean = np.mean(particles)
    posterior_std = np.std(particles)
    print(f"   Non-Parametric Posterior Mean: {posterior_mean:.6f}")
    print(f"   Posterior Uncertainty (Particle Spread): {posterior_std:.6f}")

    print("\n2. Structural Regularity Test (Ramanujan-Petersson analogy):")
    violations = chow_motive_of_evidence(y)
    print(f"   Bound Violations: {violations}")
    if violations == 0:
        print("   OBSERVATION: Effect distribution satisfies regularity bounds.")
        print("   The distribution has low spectral irregularity.")
    else:
        print("   OBSERVATION: Effect distribution exceeds regularity bounds.")
        print("   The distribution has high spectral irregularity (heavy tails or multimodality).")

    # Neural SDE (Conceptual)
    print("\n3. Evidence Drift Model (Neural SDE analogy):")
    if 'year' in data.columns:
        y_sorted = data.sort_values('year')['log_or'].values
    else:
        y_sorted = np.sort(y)
    drift = np.mean(np.diff(y_sorted))
    diffusion = np.std(np.diff(y_sorted))
    future_prediction = posterior_mean + drift * 1.0
    print(f"   Estimated Drift: {drift:.6f}, Diffusion: {diffusion:.6f}")
    print(f"   Projected effect (1 step): {future_prediction:.6f}")
    print("   NOTE: Drift model is exploratory; temporal ordering may be approximate.")

    print("\n===================================================================")
    print("  EXPLORATORY TIER COMPLETE.")
    print("===================================================================")


if __name__ == '__main__':
    main()
