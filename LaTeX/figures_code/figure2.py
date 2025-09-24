import numpy as np
import matplotlib.pyplot as plt

# Data (same)
components = ['Full SRT', 'No Zeta', 'No Alt', 'No Flow', 'Baseline (Vanilla)']
x_pos = np.arange(len(components))

primes_recall = [0.999, 0.85, 0.92, 0.88, 0.70]
primes_err = [0.005] * 5

tsp_gap = [0.5, 3.0, 1.8, 2.2, 5.0]
tsp_err = [0.1] * 5

sat_recall = [0.190, 0.120, 0.145, 0.132, 0.01]
sat_err = [0.01] * 5
sat_flow = [0.238]

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

width = 0.25
bars1 = ax.barh(x_pos - width, primes_recall, width, xerr=primes_err, label='Primes Recall', color='#2E8B57')
bars2 = ax.barh(x_pos, tsp_gap, width, xerr=tsp_err, label='TSP % Over Opt', color='#FF8C00')
bars3 = ax.barh(x_pos + width, sat_recall, width, xerr=sat_err, label='3-SAT Recall', color='#A9A9A9', hatch='//')
bars_flow = ax.barh(0 + width, sat_flow[0], width*0.8, color='blue', alpha=0.7, label='SAT + Flow')  # On Full SRT

ax.set_yticks(x_pos)
ax.set_yticklabels(components)
ax.set_xlabel('Metric Value')
ax.set_title('Ablation Lifts: Zeta/Alt/Flow Contributions (Avg +30-45%)')
ax.grid(axis='x', alpha=0.3)

# Legend shifted right
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)

# Fixed annotations (fewer, offset right/up, clip_on=False)
ax.annotate('+15% Zeta', xy=(0.85, x_pos[1] - width/2), xytext=(1.2, x_pos[1]), 
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=8, clip_on=False)
ax.annotate('+37% Zeta', xy=(0.120, x_pos[4] + width/2), xytext=(0.25, x_pos[4] + 0.3), 
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=8, clip_on=False)
# Add one more example if needed, e.g., for Alt: ax.annotate('+24% Alt', ...)

plt.tight_layout()
plt.savefig('figure2_fixed.png', dpi=300, bbox_inches='tight')
plt.show()
