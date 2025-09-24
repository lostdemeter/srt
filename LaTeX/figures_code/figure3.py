import numpy as np
import matplotlib.pyplot as plt

# Data (same)
delta = np.arange(1, 11)
gamma_theory = 1 / (3 * delta)
emp_delta = [1, 2, 4]
emp_gamma = [0.33, 0.16, 0.12]
gamma_extrap = np.maximum(1/(3*delta), 0.01)
threshold = 1/9

# Plot
fig, ax = plt.subplots(figsize=(5, 4))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

ax.plot(delta, gamma_theory, 'b-', linewidth=2, label='Theory (Lem 6)')
ax.scatter(emp_delta, emp_gamma, c='red', s=100, zorder=5, label='Empirical')
ax.plot(delta, gamma_extrap, 'orange', ls='--', linewidth=2, label='Extrapolation')
ax.axvspan(6, 10, alpha=0.3, color='red', label='Failure (γ<1/9)')
ax.axhline(threshold, color='red', ls=':', linewidth=1.5, label='Threshold')

ax.set_xlabel('Δ (Regularity)')
ax.set_ylabel('Spectral Gap γ')
ax.set_title('Spectral Gap γ vs. Δ: Viability to Δ=6 (nc-Flow)')
ax.set_xticks(delta)
ax.grid(alpha=0.3)

# Legend upper right, compact
ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8)

# Fixed annotations (Primes lowered/left; adversarial lowered)
labels = ['Primes', 'TSP', 'SAT']
for i, (d, g) in enumerate(zip(emp_delta, emp_gamma)):
    offset_x = 0.15 if i == 0 else 0.25  # Pull Primes left more
    offset_y = 0.005 if i == 0 else 0.01  # Lower Primes y
    ax.annotate(labels[i], (d, g), xytext=(d + offset_x, g + offset_y), fontsize=8)
ax.annotate('Adversarial blowup:\nFull scan needed', xy=(9.5, 0.04), xytext=(7.5, 0.20), 
            arrowprops=dict(arrowstyle='->', color='orange'), fontsize=8, ha='left', clip_on=False)

plt.tight_layout()
plt.savefig('figure3_fixed.png', dpi=300, bbox_inches='tight')
plt.show()
