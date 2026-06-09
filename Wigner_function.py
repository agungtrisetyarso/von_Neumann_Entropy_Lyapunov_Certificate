!pip install qutip matplotlib -q

from qutip import *
import matplotlib.pyplot as plt

# Assuming 'rho' is your final/iterative density matrix (dim=4 or 5)
xvec = np.linspace(-5, 5, 200)
W = wigner(rho, xvec, xvec)

plt.figure(figsize=(8, 6))
plt.contourf(xvec, xvec, W, 100, cmap='RdBu')
plt.colorbar(label='Wigner Function W(q,p)')
plt.xlabel('q')
plt.ylabel('p')
plt.title('Wigner Distribution at Convergence\n(Peak indicates pure feasible allocation)')
plt.grid(True, alpha=0.3)
plt.show()
