# ================================================
# COMPLETE: Initial vs Final Wigner Distribution
# Side-by-Side for Paper (Fixed)
# ================================================

!pip install qutip matplotlib numpy scipy -q

import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm

def sinkhorn_projection(A, tol=1e-9, max_iter=500):
    A = np.abs(A).astype(float) + 1e-12
    for _ in range(max_iter):
        A /= np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        A /= np.maximum(A.sum(axis=0, keepdims=True), 1e-12)
    return A

# Setup
dim = 5
np.random.seed(42)

# Initial mixed state (high entropy)
rho_init_mat = np.random.rand(dim, dim)
rho_initial = Qobj(sinkhorn_projection(rho_init_mat))

# Run short simulation to get converged state
alpha = 0.05
beta = 0.4
n_steps = 300
rho = rho_initial.copy()

for k in range(n_steps):
    rho_full = rho.full()
    log_rho = logm(rho_full + 1e-12 * np.eye(dim))
    purity_term = (rho**2) / max((rho**2).tr().real, 1e-8)
    temp = expm(-alpha * log_rho) @ purity_term.full() @ expm(-alpha * log_rho).conj().T
    rho_new = Qobj(sinkhorn_projection(temp))
    rho = (1 - beta) * rho_new + beta * rho
    rho = Qobj(sinkhorn_projection(rho.full()))

rho_final = rho

# Wigner Plots - Side by Side
xvec = np.linspace(-5, 5, 200)

W_init = wigner(rho_initial, xvec, xvec)
W_final = wigner(rho_final, xvec, xvec)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

cf0 = axs[0].contourf(xvec, xvec, W_init, 100, cmap='RdBu')
axs[0].set_title('Wigner Distribution\nInitial Mixed State (High Entropy)')
axs[0].set_xlabel('q')
axs[0].set_ylabel('p')
plt.colorbar(cf0, ax=axs[0])

cf1 = axs[1].contourf(xvec, xvec, W_final, 100, cmap='RdBu')
axs[1].set_title('Wigner Distribution\nAt Convergence (Pure Feasible Allocation)')
axs[1].set_xlabel('q')
plt.colorbar(cf1, ax=axs[1])

plt.suptitle('Von Neumann Entropy Decrease → Phase-Space Localization', fontsize=14)
plt.tight_layout()
plt.show()

# Metrics
print("Initial Entropy:", entropy_vn(rho_initial))
print("Final Entropy:", entropy_vn(rho_final))
print("Final Purity:", (rho_final**2).tr().real)
