from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from numpy import pi
import matplotlib.pyplot as plt
import numpy as np

def quantum_phase_estimation():
   
    num_estimation_qubits = 4
    target_qubit = 1
    total_qubits = num_estimation_qubits + target_qubit

    qc = QuantumCircuit(total_qubits, num_estimation_qubits)

    qc.h(range(num_estimation_qubits))

    qc.x(num_estimation_qubits)

    theta = 0.3125  
    for i in range(num_estimation_qubits):
        angle = 2 * pi * theta * (2 ** i)
 
    qc.append(QFT(num_estimation_qubits, inverse=True, do_swaps=True).to_gate(), range(num_estimation_qubits))

    qc.measure(range(num_estimation_qubits), range(num_estimation_qubits))

    backend = Aer.get_backend("qasm_simulator")
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=1024).result()
    counts = result.get_counts()

    print("\n🌀 QPE Result - Simulated Disaster Severity")
    print("Estimated Phase Histogram:")
    
    print(counts)


    plot_histogram(counts)

    plt.title("Quantum Phase Estimation Output (Disaster Magnitude)")
    plt.show()

    return counts


if __name__ == "__main__":
    quantum_phase_estimation()