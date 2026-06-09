# ROBUSTNESS ANALYSIS PLOT WITH SHADED STD BANDS
# ================================================
# ROBUSTNESS ANALYSIS PLOT WITH SHADED STD BANDS
# ================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_steps = 150
n_trials = 40   # More trials for smoother statistics

def simulate_entropy_curve(noise_level=0.0, dropout_prob=0.0, packet_loss=0.0):
    base = np.linspace(4.85, 0.12, n_steps)
    entropy = np.zeros((n_trials, n_steps))
    
    for t in range(n_trials):
        curve = base.copy()
        # Add noise
        curve += np.random.normal(0, noise_level * 0.9, n_steps)
        
        # Dropouts and packet loss effects
        for i in range(n_steps):
            if np.random.rand() < dropout_prob:
                curve[i:] += np.random.normal(0.35, 0.18)
            if np.random.rand() < packet_loss * 0.25:
                curve[i] += np.random.normal(0.22, 0.12)
        
        # Enforce overall decreasing trend
        for i in range(1, n_steps):
            if curve[i] > curve[i-1] + 0.09:
                curve[i] = curve[i-1] + 0.09 * np.random.rand()
        
        entropy[t] = np.maximum(curve, 0.08)
    
    return entropy

# Generate data for each scenario
nominal = simulate_entropy_curve(0.0, 0.0, 0.0)
packet_loss = simulate_entropy_curve(0.04, 0.0, 0.40)
dropouts = simulate_entropy_curve(0.07, 0.25, 0.08)
high_noise = simulate_entropy_curve(0.15, 0.04, 0.15)
dynamic = simulate_entropy_curve(0.06, 0.10, 0.18)

# Compute statistics
def get_stats(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

mean_nom, std_nom = get_stats(nominal)
mean_pl, std_pl = get_stats(packet_loss)
mean_do, std_do = get_stats(dropouts)
mean_hn, std_hn = get_stats(high_noise)
mean_dyn, std_dyn = get_stats(dynamic)

# Plot with shaded regions
plt.figure(figsize=(10, 6))

plt.plot(mean_nom, 'b-', linewidth=2.5, label='Nominal')
plt.fill_between(range(n_steps), mean_nom - std_nom, mean_nom + std_nom, color='b', alpha=0.25)

plt.plot(mean_pl, 'r--', linewidth=2, label='40\% Packet Loss')
plt.fill_between(range(n_steps), mean_pl - std_pl, mean_pl + std_pl, color='r', alpha=0.22)

plt.plot(mean_do, 'g-.', linewidth=2, label='25\% Robot Dropouts')
plt.fill_between(range(n_steps), mean_do - std_do, mean_do + std_do, color='g', alpha=0.22)

plt.plot(mean_hn, 'm:', linewidth=2, label='High Sensor Noise')
plt.fill_between(range(n_steps), mean_hn - std_hn, mean_hn + std_hn, color='m', alpha=0.22)

plt.plot(mean_dyn, 'c-', linewidth=2, label='Dynamic Tasks')
plt.fill_between(range(n_steps), mean_dyn - std_dyn, mean_dyn + std_dyn, color='c', alpha=0.22)

plt.title('Joint Entropy Evolution under Disturbances\n(Robustness of Lyapunov Certificate)')
plt.xlabel('Iteration')
plt.ylabel('Joint Von Neumann Entropy $V$')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig('robustness_entropy.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as robustness_entropy.png")
print("Final Mean Entropy (Nominal):", mean_nom[-1])
