# ================================================
# Entropy as Lyapunov Certificate Simulation
# Von Neumann Entropy Decrease → Convergence
# For multi-agent / resource allocation (T-RO/IJRR style)
# ================================================

!pip install qutip matplotlib numpy -q

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.linalg import logm

# ----------------- Helper Functions -----------------
def von_neumann_entropy(rho, base=np.e):
    """Compute von Neumann entropy S(rho) = -Tr(rho log rho)"""
    return entropy_vn(rho, base=base)

def trace_distance(rho1, rho2):
    """Trace distance between two density matrices"""
    return 0.5 * (rho1 - rho2).norm('tr')

# ----------------- Simulation Setup -----------------
# Simple model: 3 "tasks" or basis states (Hilbert dim=3)
# Feasible allocations = pure states (deterministic assignment)
dim = 3
target_pure = fock_dm(dim, 0)  # Desired feasible allocation (pure state)

# Initial state: mixed / uncertain allocation
rho = rand_dm(dim, density=0.8)  # Random mixed state
print("Initial von Neumann Entropy:", von_neumann_entropy(rho))

# Parameters for dynamics (entropy-regularized gradient flow like update)
alpha = 0.15   # Step size / decrease rate
n_steps = 100
entropies = [von_neumann_entropy(rho)]
distances = [trace_distance(rho, target_pure)]
feasible_errors = []

# ----------------- Main Simulation Loop -----------------
for k in range(n_steps):
    # Simulate control update: drive towards lower entropy (more deterministic)
    # Example: simple dissipative dynamics + projection-like term
    H = -logm(rho.full() + 1e-10 * np.eye(dim))  # "Natural" potential
    H = Qobj(H)
    
    # Lindblad-like or gradient flow step towards purity
    drho = -1j * (H * rho - rho * H) * 0.01 + alpha * (target_pure - rho)
    rho = rho + drho * 0.1
    rho = rho / rho.tr()  # Ensure trace=1
    
    # Record metrics
    S = von_neumann_entropy(rho)
    d = trace_distance(rho, target_pure)
    entropies.append(S)
    distances.append(d)
    
    if k % 20 == 0:
        print(f"Step {k:3d} | Entropy: {S:.4f} | Dist to feasible: {d:.4f}")

# ----------------- Plot Results -----------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(entropies, 'b-', linewidth=2, label='Von Neumann Entropy S(ρ_k)')
plt.xlabel('Iteration k')
plt.ylabel('Entropy')
plt.title('Monotonic Entropy Decrease (Lyapunov Certificate)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(distances, 'r-', linewidth=2, label='Trace Distance to Feasible Set')
plt.xlabel('Iteration k')
plt.ylabel('Distance d(ρ_k, ℱ)')
plt.title('Convergence to Feasible Allocation')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("\nFinal Entropy:", entropies[-1])
print("Final Distance to Feasible Pure State:", distances[-1])
print("Monotonic decrease confirmed:", all(entropies[i] >= entropies[i+1] - 1e-6 for i in range(len(entropies)-1)))
