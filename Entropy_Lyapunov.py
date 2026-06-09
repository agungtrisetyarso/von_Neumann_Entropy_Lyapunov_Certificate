# ================================================
# FINAL MONTE CARLO - Constrained Entropy Lyapunov
# ================================================

!pip install qutip matplotlib numpy scipy -q

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm

def von_neumann_entropy(rho):
    return entropy_vn(rho, base=np.e)

def sinkhorn_projection(A, tol=1e-9, max_iter=1000):
    A = np.abs(A).astype(float) + 1e-12
    for _ in range(max_iter):
        A /= np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        A /= np.maximum(A.sum(axis=0, keepdims=True), 1e-12)
    return A

# Parameters
dim = 5
n_trials = 30
n_steps = 400
alpha = 0.05
beta = 0.4

all_entropies = []
all_purities = []

for trial in range(n_trials):
    np.random.seed(trial)
    rho = rand_dm(dim, density=0.75)
    rho = Qobj(sinkhorn_projection(rho.full()))
    
    entropies = [von_neumann_entropy(rho)]
    purities = [(rho**2).tr().real]
    
    for k in range(n_steps):
        rho_full = rho.full()
        log_rho = logm(rho_full + 1e-12 * np.eye(dim))
        purity_term = (rho**2) / max((rho**2).tr().real, 1e-8)
        temp = expm(-alpha * log_rho) @ purity_term.full() @ expm(-alpha * log_rho).conj().T
        rho_new = Qobj(sinkhorn_projection(temp))
        
        rho = (1 - beta) * rho_new + beta * rho
        rho = Qobj(sinkhorn_projection(rho.full()))
        
        entropies.append(von_neumann_entropy(rho))
        purities.append((rho**2).tr().real)
    
    all_entropies.append(entropies)
    all_purities.append(purities)

all_entropies = np.array(all_entropies)
all_purities = np.array(all_purities)

# Plots with statistics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.mean(all_entropies, axis=0), 'b-', linewidth=2.5, label='Mean Entropy')
plt.fill_between(range(n_steps+1), 
                 np.mean(all_entropies, axis=0) - np.std(all_entropies, axis=0),
                 np.mean(all_entropies, axis=0) + np.std(all_entropies, axis=0), alpha=0.3, color='b')
plt.title('Monte Carlo: Monotonic Entropy Decrease')
plt.xlabel('Iteration k')
plt.ylabel('S(ρ_k)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.mean(all_purities, axis=0), 'r-', linewidth=2.5, label='Mean Purity')
plt.fill_between(range(n_steps+1),
                 np.mean(all_purities, axis=0) - np.std(all_purities, axis=0),
                 np.mean(all_purities, axis=0) + np.std(all_purities, axis=0), alpha=0.3, color='r')
plt.title('Monte Carlo: Convergence to Pure Feasible Allocation')
plt.xlabel('Iteration k')
plt.ylabel('Purity Tr(ρ²)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

print("Final mean entropy:", np.mean(all_entropies[:,-1]))
print("Final mean purity:", np.mean(all_purities[:,-1]))
