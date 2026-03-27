"""
Hawking radiation quantum simulation
====================================

WHAT THIS SIMULATION DOES:
  We model a black hole emitting Hawking radiation using qubits. In the
  standard picture, radiation is produced in entangled pairs: one partner
  escapes to infinity ("exterior") and one falls into the hole ("interior").
  Each pair is created in a maximally entangled state (Bell pair). As more
  pairs are emitted, the exterior radiation becomes entangled with the
  interior, so the entropy of the exterior subsystem grows — this is the
  origin of the black-hole information paradox (entropy keeps growing even
  after the hole evaporates).

  The code implements two models:
  1. Time evolution: build 1, 2, ..., N pairs, evolve each system for T
     timesteps, then compute the von Neumann entropy of the exterior qubits
     only (partial trace over interior). Result: entropy vs number of pairs.
  2. Continuous emission: at each timestep, with some probability, a new
     entangled pair is added; the full system is evolved. Result: entropy
     vs time as radiation accumulates.

WHAT THE RESULTS MEAN:
  - Entropy (y-axis) is in bits (base-2 von Neumann entropy). It measures
    how much the exterior radiation is entangled with the interior (and
    thus "how much information" is hidden in the hole from the outside).
  - In the first plot: entropy generally increases with the number of pairs,
    roughly consistent with a growing "radiation entropy" as the hole
    emits more quanta.
  - In the second plot: entropy fluctuates and grows over time as new pairs
    are randomly added and the system is evolved (chaotic mixing).
  - The exact curves depend on the unitary evolution (rotation angles and
    CNOTs); the important qualitative result is the growth of exterior
    entropy with more radiation, as in Hawking's calculation.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np
import matplotlib.pyplot as plt


# Create n pairs of Hawking radiation particles (exterior + interior qubits per pair)
def generate_pairs(n_pairs):
    # Create circuit of 2n qubits
    qc = QuantumCircuit(2 * n_pairs)

    # Iterate through pairs, entangle exterior and interior qubit
    for i in range(n_pairs):
        ext = 2 * i
        int = ext + 1

        # Superposition
        qc.h(ext)
        # Entanglement
        qc.cx(ext, int)

    return qc


# Apply one timestep of unitary evolution using a log-depth butterfly pattern.
# Each round doubles the interaction distance, so after ceil(log2(n)) rounds every
# pair of qubits has been connected — full scrambling in O(log n) rounds vs O(n)
# for the old nearest-neighbour chain. This matches the "fast scrambler" conjecture
# for black holes (Sekino-Susskind: BHs scramble in O(log S) time).
def evolve_radiation(qc, n_pairs, timestep):
    # Single-qubit rotations on exterior qubits (unchanged)
    for i in range(0, 2 * n_pairs, 2):
        angle = 0.1 * timestep
        qc.rz(angle, i)
        qc.ry(angle, i)

    if n_pairs < 2:
        return qc

    rounds = int(np.ceil(np.log2(n_pairs)))

    # Butterfly scrambling within exterior subsystem.
    # Round r pairs logical index i with i XOR 2^r (doubling reach each round).
    for r in range(rounds):
        step = 1 << r  # 2^r
        for i in range(n_pairs):
            j = i ^ step
            if j > i and j < n_pairs:
                qc.cx(2 * i, 2 * j)  # ext_i → ext_j
                qc.rz(0.2 * np.random.random(), 2 * i)

    # Butterfly scrambling crossing the exterior/interior boundary.
    # ext_i is paired with int_j using the same butterfly indexing, so
    # entanglement spreads across the subsystem boundary in O(log n) rounds.
    for r in range(rounds):
        step = 1 << r
        for i in range(n_pairs):
            j = i ^ step
            if j > i and j < n_pairs:
                qc.cx(2 * i, 2 * j + 1)  # ext_i → int_j
                qc.rz(0.15 * np.random.random(), 2 * j + 1)

    return qc


# For n = 1, 2, ..., max_pairs: create n entangled pairs, evolve for timesteps,
# then compute von Neumann entropy of the exterior subsystem (partial trace over interior).
# Returns list of entropies — one per n — so we can plot entropy vs number of pairs.
def time_evolution(max_pairs, timesteps):
    entropies = []

    # Generate pairs up to max pairs
    for n in range(1, max_pairs + 1):
        qc = generate_pairs(n)

        # Iterate through timesteps and evolve system
        for t in range(timesteps):
            qc = evolve_radiation(qc, n, t)

        # Get state, trace out interior to get exterior subsystem, measure entropy
        state = Statevector(qc)
        interior_qubits = [2 * i + 1 for i in range(n)]  # Trace out interior
        ext_state = partial_trace(state, interior_qubits)
        entropy_val = entropy(ext_state, base=2)  # Don't shadow function name

        entropies.append(entropy_val)

    return entropies


# Stochastically add new entangled pairs over time (each step: add a pair with probability
# emission_rate), evolve the full system each step, and record exterior entropy at each time.
# Returns list of entropies over total_time steps — models entropy growth under continuous emission.
def continuous_emission_model(total_time, emission_rate):
    entropies = []
    qc = QuantumCircuit(0)

    # Operate for given time
    for t in range(total_time):
        # Randomly add new pair of qubits to circuit
        if np.random.random() < emission_rate:
            new_qc = QuantumCircuit(qc.num_qubits + 2)
            new_qc.compose(qc, range(qc.num_qubits), inplace=True)

            # Establish exterior and interior qubit
            ext = qc.num_qubits
            int = ext + 1

            # Entangle pair
            new_qc.h(ext)
            new_qc.cx(ext, int)

            qc = new_qc

        # Evolve system
        if qc.num_qubits > 0:
            qc = evolve_radiation(qc, qc.num_qubits // 2, t)

        # Measure entropy
        if qc.num_qubits > 0:
            state = Statevector(qc)
            n_pairs = qc.num_qubits // 2
            interior_qubits = [2 * i + 1 for i in range(n_pairs)]
            ext_state = partial_trace(state, interior_qubits)
            entropy_val = entropy(ext_state, base=2)

            entropies.append(entropy_val)
        else:
            entropies.append(0)

    return entropies


# Page curve model: start with a fixed-size black hole (bh_size qubits), apply heavy
# scrambling, then "evaporate" one qubit at a time. At step k, qubits 0..k-1 are
# radiation and qubits k..bh_size-1 are the remaining black hole. The entropy of the
# radiation follows the Page curve: S(k) ≈ min(k, N-k).
def page_curve_model(bh_size, scramble_steps=3):
    qc = QuantumCircuit(bh_size)

    # Initialize all qubits in superposition
    for i in range(bh_size):
        qc.h(i)

    # Heavy initial scrambling to create a highly entangled pure state
    rounds = int(np.ceil(np.log2(bh_size)))
    for _ in range(scramble_steps):
        for r in range(rounds):
            step = 1 << r
            for i in range(bh_size):
                j = i ^ step
                if j > i and j < bh_size:
                    qc.cx(i, j)
                    qc.rz(np.random.random(), i)
                    qc.ry(np.random.random(), j)

    entropies = []

    # Evaporate one qubit at a time
    for k in range(1, bh_size):
        # Radiation: qubits 0..k-1, Black hole: qubits k..bh_size-1
        bh_remaining = bh_size - k

        # Scramble the remaining BH interior after each emission
        if bh_remaining >= 2:
            bh_rounds = int(np.ceil(np.log2(bh_remaining)))
            for r in range(bh_rounds):
                step = 1 << r
                for i in range(bh_remaining):
                    j = i ^ step
                    if j > i and j < bh_remaining:
                        qc.cx(k + i, k + j)
                        qc.rz(np.random.random(), k + i)
                        qc.ry(np.random.random(), k + j)

        # Measure entropy of radiation by tracing out the BH
        state = Statevector(qc)
        bh_qubits = list(range(k, bh_size))
        radiation_dm = partial_trace(state, bh_qubits)
        entropy_val = entropy(radiation_dm, base=2)
        entropies.append(entropy_val)

    return entropies


if __name__ == "__main__":
    # --- Fixed seed for reproducible graphs and tangible results ---
    np.random.seed(100)

    # ========== Model 1: Entropy vs number of entangled pairs ==========
    # More pairs => more exterior–interior entanglement => higher exterior entropy.
    max_pairs = 8
    timesteps = 5
    entropies_vs_pairs = time_evolution(max_pairs, timesteps)
    n_pairs_list = list(range(1, max_pairs + 1))

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    # --- 1. Plot the original simulated data ---
    ax1.plot(
        n_pairs_list,
        entropies_vs_pairs,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=8,
        label="Simulated Entropy",
    )

    # --- 2. ADD THIS: Calculate and plot the line of best fit ---
    m, c = np.polyfit(n_pairs_list, entropies_vs_pairs, 1)
    # Use polyval to generate the line data based on our fit coefficients
    ax1.plot(
        n_pairs_list,
        np.polyval([m, c], n_pairs_list),
        color="crimson",
        linestyle="--",
        label=f"Linear Fit (slope={m:.2f})",
    )

    # --- 3. Formatting (Ensure ax1.legend() is called) ---
    ax1.set_xlabel("Number of Hawking radiation pairs", fontsize=11)
    ax1.set_ylabel("Exterior entropy (bits)", fontsize=11)
    ax1.set_title("Exterior entropy vs number of emitted pairs (evolved system)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_pairs_list)
    ax1.legend()  # This displays the labels on the graph
    fig1.savefig("hawking_entropy_vs_pairs.png", dpi=150, bbox_inches="tight")

    # ========== Model 2: Entropy vs time (continuous stochastic emission) ==========
    total_time = 30
    emission_rate = 0.25  # probability per timestep to emit a new pair
    entropies_vs_time = continuous_emission_model(total_time, emission_rate)
    time_steps = list(range(total_time))
    print("Debug: ", len(time_steps), len(entropies_vs_time))
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    # --- 1. Plot the original stochastic data ---
    ax2.plot(
        time_steps,
        entropies_vs_time,
        color="darkgreen",
        alpha=0.5,
        linewidth=1.2,
        label="Stochastic Emission",
    )

    # --- 2. Calculate and plot the line of best fit ---
    # We use time_steps as X and entropies_vs_time as Y
    m2, c2 = np.polyfit(time_steps, entropies_vs_time, 1)
    ax2.plot(
        time_steps,
        np.polyval([m2, c2], time_steps),
        color="orange",
        linestyle="--",
        linewidth=2,
        zorder=5,
        label=f"Trendline (slope={m2:.3f})",
    )

    # --- 3. Formatting ---
    ax2.set_xlabel("Time step", fontsize=11)
    ax2.set_ylabel("Exterior entropy (bits)", fontsize=11)
    ax2.set_title(f"Exterior entropy over time (rate={emission_rate})")
    ax2.grid(True, alpha=0.3)
    ax2.legend()  # Crucial to see the trendline label
    fig2.savefig("hawking_entropy_vs_time.png", dpi=150, bbox_inches="tight")
    print(f"slope={m2:.6f}, intercept={c2:.6f}")
    print(entropies_vs_time)
    # ========== Model 3: Page curve ==========
    bh_size = 12
    page_entropies = page_curve_model(bh_size)
    emission_steps = list(range(1, bh_size))  # k = 1, 2, ..., N-1

    # Theoretical Page curve: S(k) = min(k, N-k)
    page_theoretical = [min(k, bh_size - k) for k in emission_steps]

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(
        emission_steps,
        page_entropies,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=6,
        label="Simulated Page Curve",
    )
    ax3.plot(
        emission_steps,
        page_theoretical,
        "s--",
        color="crimson",
        linewidth=1.5,
        markersize=5,
        alpha=0.7,
        label="Theoretical: min(k, N−k)",
    )
    ax3.axvline(
        x=bh_size / 2, color="gray", linestyle=":", alpha=0.5, label="Page time (N/2)"
    )
    ax3.set_xlabel("Qubits emitted (k)", fontsize=11)
    ax3.set_ylabel("Radiation entropy (bits)", fontsize=11)
    ax3.set_title(f"Page Curve (N={bh_size} qubit black hole)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.savefig("hawking_page_curve.png", dpi=150, bbox_inches="tight")

    # ========== Summary statistics (tangible results) ==========
    print("=" * 60)
    print("HAWKING RADIATION SIMULATION — TANGIBLE RESULTS")
    print("=" * 60)
    print("\n--- Model 1: Entropy vs number of pairs ---")
    print(f"  Pairs: {n_pairs_list}")
    print(f"  Entropy (bits): {[round(s, 4) for s in entropies_vs_pairs]}")
    print(
        f"  Max exterior entropy: {max(entropies_vs_pairs):.4f} bits (at {n_pairs_list[entropies_vs_pairs.index(max(entropies_vs_pairs))]} pairs)"
    )
    print(f"  Min exterior entropy: {min(entropies_vs_pairs):.4f} bits (at 1 pair)")

    print("\n--- Model 2: Continuous emission over time ---")
    valid_ent = [e for e in entropies_vs_time if e > 0]
    print(f"  Time steps: 0..{total_time - 1}, emission rate: {emission_rate}")
    if valid_ent:
        print(f"  Final exterior entropy: {entropies_vs_time[-1]:.4f} bits")
        print(f"  Mean entropy (when > 0): {np.mean(valid_ent):.4f} bits")
        print(f"  Max entropy in run: {max(entropies_vs_time):.4f} bits")

    print(f"\n--- Model 3: Page curve (N={bh_size}) ---")
    print(f"  Emitted: {emission_steps}")
    print(f"  Entropy (bits): {[round(s, 4) for s in page_entropies]}")
    peak_k = emission_steps[page_entropies.index(max(page_entropies))]
    print(
        f"  Peak entropy: {max(page_entropies):.4f} bits at k={peak_k} (Page time ≈ {bh_size // 2})"
    )
    print(f"  Final entropy (k={bh_size - 1}): {page_entropies[-1]:.4f} bits")
    print(
        "\n  Plots saved: hawking_entropy_vs_pairs.png, hawking_entropy_vs_time.png, hawking_page_curve.png"
    )
    print("=" * 60)
