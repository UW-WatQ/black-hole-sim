from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
import numpy as np
import matplotlib.pyplot as plt

# Create n pairs of Hawking radiation particles
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
        qc.cx (ext, int)

    return qc

# Apply one timestep of evolution to qubits
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

# Incrementally create pairs up to specified max and evolve system by specified timesteps
def time_evolution(max_pairs, timesteps):
    entropies = []

    # Generate pairs up to max pairs
    for n in range(1, max_pairs + 1):
        qc = generate_pairs(n)

        # Iterate through timesteps and evolve system
        for t in range(timesteps):
            qc = evolve_radiation(qc, n, t)

        # Get state, separate out exterior qubits, measure entropy of qubits
        state = Statevector(qc)
        keep_qubits = [i * 2 for i in range(n)]
        ext_state = partial_trace(state, keep_qubits)
        entropy = entropy(ext_state, base=2)

        entropies.append(entropy)

    return entropies

# Model continuous emission of Hawking radiation
def continuous_emission_model(total_time, emission_rate):
    entropies = []
    qc = QuantumCircuit(0)

    # Operate for given time
    for t in range(total_time):
        # Randomly add new pair of qubits to circuit
        if (np.random.random() < emission_rate):
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
        if (qc.num_qubits > 0):
            qc = evolve_radiation(qc, qc.num_qubits // 2, t)

        # Measure entropy
        if (qc.num_qubits > 0):
            state = Statevector(qc)
            n_pairs = qc.num_qubits // 2
            keep = [2 * i for i in range(n_pairs)]
            ext_state = partial_trace(state, keep)
            entropy = entropy(ext_state, base=2)

            entropies.append(entropy)
        else:
            entropies.append(0)

    return entropies