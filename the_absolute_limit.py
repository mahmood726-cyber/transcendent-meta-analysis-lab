import numpy as np

from core.data_loader import load_integrated_df


def chaitins_omega_approximation(y, num_bits=32):
    """
    Maps the sign of clinical effect sizes to a binary sequence and
    computes a weighted sum analogous to Chaitin's Omega.
    NOTE: This is an analogy -- the true Omega is a property of a
    specific universal Turing machine and is provably uncomputable.
    """
    binary_evidence = (y[:num_bits] < 0).astype(int)

    omega_approx = 0.0
    for i, bit in enumerate(binary_evidence):
        omega_approx += bit * (2 ** -(i + 1))

    return binary_evidence, omega_approx


def turing_degree_of_medicine(binary_seq):
    """
    Lempel-Ziv 76 complexity: count distinct subsequences when parsing
    the binary string left-to-right.  Each time the current extension
    is not a previously seen substring, a new word is emitted.
    Normalized by n/log2(n) so the result falls in [0, 1] for
    comparison against the random baseline.
    """
    n = len(binary_seq)
    if n == 0:
        return 0.0
    s = "".join(str(b) for b in binary_seq)
    words = set()
    complexity = 0
    start = 0
    while start < n:
        end = start + 1
        while end <= n and s[start:end] in words:
            end += 1
        words.add(s[start:end])
        complexity += 1
        start = end
    # Normalize: random binary string has LZ ~ n / log2(n)
    normalizer = n / np.log2(n) if n > 1 else 1.0
    return complexity / normalizer


def main():
    print("===================================================================")
    print("  EXPLORATORY: ALGORITHMIC COMPLEXITY OF CLINICAL EVIDENCE")
    print("===================================================================")

    # Load Data
    try:
        df = load_integrated_df(require_se=False)
    except (FileNotFoundError, ValueError) as e:
        print(f"Could not load data: {e}")
        return

    y = df['log_or'].values

    print("\n1. Binary Encoding of Effect Directions:")
    print("   Encoding each trial as 1 (beneficial, log_or < 0) or 0 (non-beneficial).")

    bits, omega = chaitins_omega_approximation(y, num_bits=64)
    binary_str = "".join([str(b) for b in bits])

    print(f"   Evidence Binary Tape (First 64 trials): {binary_str}")
    print(f"   Weighted sum (Omega analogy): {omega:.15f}")
    print(f"   Proportion beneficial: {np.mean(bits):.4f}")

    print("\n2. Lempel-Ziv Complexity (Algorithmic Depth):")
    lz_complexity = turing_degree_of_medicine(bits)
    print(f"   Normalized LZ Complexity: {lz_complexity:.4f}")
    print(f"   (Values near 1.0 suggest high randomness; near 0 suggest compressibility)")

    if lz_complexity > 0.5:
        print("   OBSERVATION: The binary evidence sequence has high algorithmic complexity.")
        print("   This suggests the sequence of beneficial/non-beneficial outcomes is not")
        print("   easily compressible -- consistent with genuine uncertainty in drug efficacy.")
    else:
        print("   OBSERVATION: The binary evidence sequence is algorithmically compressible.")
        print("   This suggests a systematic pattern in the direction of clinical effects.")

    print("\n   CAVEAT: 64 bits is too short for reliable randomness classification.")
    print("   Normalized LZ complexity of short sequences has high variance.")
    print("   These results are exploratory and should not be over-interpreted.")

    print("\n===================================================================")
    print("  EXPLORATORY ANALYSIS COMPLETE.")
    print("===================================================================")


if __name__ == '__main__':
    main()
