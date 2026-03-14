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


# Apply one timestep of unitary evolution: rotate exterior qubits and couple nearest neighbours
# to mimic chaotic dynamics (scrambling) of the radiation field.
def evolve_radiation(qc, n_pairs, timestep):
    # Evolve exterior particles (even)
    for i in range(0, 2 * n_pairs, 2):
        # Rotate about z and y axes by angle scaling with timestep
        angle = 0.1 * timestep
        qc.rz(angle, i)
        qc.ry(angle, i)

    # Interaction between particles (nearest neighbour)
    for i in range(0, 2 * n_pairs - 2, 2):
        # Simulate chaotic mixing of particles
        qc.cx(i, i + 2)
        qc.rz(0.2 * np.random.random(), i)

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


if __name__ == "__main__":
    # --- Fixed seed for reproducible graphs and tangible results ---
    np.random.seed(42)

    # ========== Model 1: Entropy vs number of entangled pairs ==========
    # More pairs => more exterior–interior entanglement => higher exterior entropy.
    max_pairs = 8
    timesteps = 5
    entropies_vs_pairs = time_evolution(max_pairs, timesteps)
    n_pairs_list = list(range(1, max_pairs + 1))

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(n_pairs_list, entropies_vs_pairs, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Hawking radiation pairs", fontsize=11)
    ax1.set_ylabel("Exterior entropy (bits)", fontsize=11)
    ax1.set_title("Exterior entropy vs number of emitted pairs (evolved system)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_pairs_list)
    fig1.tight_layout()
    fig1.savefig("hawking_entropy_vs_pairs.png", dpi=150)
    plt.close(fig1)

    # ========== Model 2: Entropy vs time (continuous stochastic emission) ==========
    total_time = 20
    emission_rate = 0.15  # probability per timestep to emit a new pair
    entropies_vs_time = continuous_emission_model(total_time, emission_rate)
    time_steps = list(range(total_time))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(time_steps, entropies_vs_time, color="darkgreen", alpha=0.8, linewidth=1.2)
    ax2.set_xlabel("Time step", fontsize=11)
    ax2.set_ylabel("Exterior entropy (bits)", fontsize=11)
    ax2.set_title("Exterior entropy over time (continuous emission, rate=0.15)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("hawking_entropy_vs_time.png", dpi=150)
    plt.close(fig2)

    # ========== Summary statistics (tangible results) ==========
    print("=" * 60)
    print("HAWKING RADIATION SIMULATION — TANGIBLE RESULTS")
    print("=" * 60)
    print("\n--- Model 1: Entropy vs number of pairs ---")
    print(f"  Pairs: {n_pairs_list}")
    print(f"  Entropy (bits): {[round(s, 4) for s in entropies_vs_pairs]}")
    print(f"  Max exterior entropy: {max(entropies_vs_pairs):.4f} bits (at {n_pairs_list[entropies_vs_pairs.index(max(entropies_vs_pairs))]} pairs)")
    print(f"  Min exterior entropy: {min(entropies_vs_pairs):.4f} bits (at 1 pair)")

    print("\n--- Model 2: Continuous emission over time ---")
    valid_ent = [e for e in entropies_vs_time if e > 0]
    print(f"  Time steps: 0..{total_time - 1}, emission rate: {emission_rate}")
    if valid_ent:
        print(f"  Final exterior entropy: {entropies_vs_time[-1]:.4f} bits")
        print(f"  Mean entropy (when > 0): {np.mean(valid_ent):.4f} bits")
        print(f"  Max entropy in run: {max(entropies_vs_time):.4f} bits")
    print("\n  Plots saved: hawking_entropy_vs_pairs.png, hawking_entropy_vs_time.png")
    print("=" * 60)
