import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Synthetic data (same as before)
np.random.seed(42)
n_sample = 10000
affinities_res = np.random.beta(20, 2, n_sample)  # Mean ~0.909
true_sols_res = np.random.normal(0.95, 0.01, 42)
true_sols_res = np.clip(true_sols_res, 0, 1)
affinities_dis = np.random.beta(2, 5, n_sample)  # Mean ~0.286

# KDEs (clip to avoid tiny values)
kde_res = gaussian_kde(affinities_res)
kde_dis = gaussian_kde(affinities_dis)
x_grid = np.linspace(0, 1, 200)
density_res = np.clip(kde_res(x_grid), 1e-3, None)  # Clip floor at 1e-3
density_dis = np.clip(kde_dis(x_grid), 1e-3, None)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9  # Smaller for less overlap

# Left: Resonant
ax1.hist(affinities_res, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='navy')
ax1.plot(x_grid, density_res, linewidth=2, color='darkblue')
for i, sol in enumerate(true_sols_res[:5]):  # Limit bars to avoid clutter; rest implied
    ax1.bar(sol, np.interp(sol, x_grid, density_res), width=0.01, color='red', alpha=0.8)
ax1.axhline(y=np.mean(density_res), color='green', ls='--', alpha=0.7)
ax1.set_title('Resonant (ΔH ≈ 0): Aligned Eig Supports (SAT Solutions)')
ax1.set_xlabel('Affinity Score')
ax1.set_ylabel('Density')
ax1.set_ylim(0, np.max(density_res) * 1.1)  # Linear, tight y
ax1.grid(alpha=0.3)

# Right: Dissonant
ax2.hist(affinities_dis, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='darkred')
ax2.plot(x_grid, density_dis, linewidth=2, color='maroon')
ax2.axvspan(0.3, 0.5, alpha=0.3, color='orange')
ax2.set_title('Dissonant (ΔH > 0): Sidelobe Interference')
ax2.set_xlabel('Affinity Score')
ax2.set_ylabel('Density')
ax2.set_ylim(0, np.max(density_dis) * 1.1)
ax2.grid(alpha=0.3)

# Annotations (non-overlapping)
ax1.text(0.02, 0.98, 'Zeta-damped overlap: Var(δ^alt) ≤ z²L/2 (Lemma 5)', 
         transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), 
         fontsize=8, va='top')
ax2.text(0.02, 0.02, 'No entrainment: γ < 1/9 threshold (failure mode)', 
         transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), 
         fontsize=8, va='bottom')

# Shared external legend (manual, no unpack error)
handles = [
    plt.Line2D([0], [0], color='darkblue', lw=2, label='Resonant KDE'),
    plt.Rectangle((0,0),1,1, color='red', alpha=0.8, label='Solutions (n=42)'),
    plt.Line2D([0], [0], color='maroon', lw=2, label='Dissonant KDE'),
    plt.Rectangle((0,0),1,1, color='orange', alpha=0.3, label='Sidelobes')
]
fig.legend(handles, ['Resonant KDE', 'Solutions (n=42)', 'Dissonant KDE', 'Sidelobes'], 
           loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=8)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Extra bottom space for legend
plt.savefig('figure1_fixed.png', dpi=300, bbox_inches='tight')
plt.show()
