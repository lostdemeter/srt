import numpy as np
import mpmath as mp
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.integrate import odeint
import time
import matplotlib.pyplot as plt

mp.mp.dps = 15
np.random.seed(42)

n_vars = 10
m_clauses = 100
z = 0.05
corr = 0.12
subsample_size = 64
use_zeta = True
use_alt = True
use_flow = True
use_rh_gct = True

clauses = []
for _ in range(m_clauses):
    lits = np.random.choice([-1, 1], 3) * np.random.randint(1, n_vars + 1, 3)
    clauses.append(tuple(lits))
print(f"Generated {m_clauses} clauses (Δ≈{m_clauses/n_vars:.1f}; sample: {clauses[0]})")

all_assigns = [tuple(np.random.randint(0, 2, n_vars)) for _ in range(subsample_size)]

affinities = np.random.rand(subsample_size)

# Pre-cache zeta zeros
zeta_zeros = [float(mp.zetazero(k).imag) for k in range(1, 11)]

def rh_guided_zeta(k, approx_order=5):
    z = mp.zetazero(k)
    mult = mp.sqrt(k) * mp.sin(mp.pi * k / approx_order)
    return float((z.imag + mult) % (2 * mp.pi))

# Pre-cache RH-GCT phases for k=1 to 10
rh_phases = [rh_guided_zeta(k) for k in range(1, 11)]

def build_ds(affinities, use_alt=True, use_zeta=True, use_rh_gct=False):
    n = len(affinities)
    data = affinities.copy()
    rows = np.arange(n, dtype=int)
    cols = rows.copy()
    for i in range(n):
        for j in range(i + 1, n):
            gamma_ij = corr * np.random.rand()
            k = ((i + j) % 10) + 1
            phase = rh_phases[k-1] if use_rh_gct and use_zeta else zeta_zeros[k-1]
            delta = ((-1)**(i + j) * gamma_ij * phase if use_alt else gamma_ij * phase)
            data = np.append(data, [delta, np.conj(delta)])
            rows = np.append(rows, [i, j])
            cols = np.append(cols, [j, i])
    ds_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))
    return ds_sparse

def hdr_smooth(eigenvals, M=3, sigma0=0.1):
    smoothed = eigenvals.real.copy()
    for m in range(M):
        eps_m = np.random.uniform(-0.02, 0.02)
        sigma_m = sigma0 + eps_m
        phi_m = np.random.uniform(0, 2 * np.pi)
        t = np.mean(smoothed)
        damp = np.exp(-sigma_m**2 * (smoothed - t)**2 + 1j * phi_m * smoothed).real / M
        smoothed += damp
    return smoothed

def flow_ode(X, t, smoothed_evals, target_gamma=1/(3*10)):
    theta, zeta_w = X
    L_loss = np.mean((smoothed_evals - np.mean(smoothed_evals))**2) + z * np.var(smoothed_evals)
    grad_theta = 2 * (theta - 0.15) + 0.1 * max(0, target_gamma - np.std(smoothed_evals)) * theta
    grad_zeta = 2 * (zeta_w - 0.06) + 0.2 * max(0, target_gamma - np.std(smoothed_evals)) * zeta_w
    G_theta = -0.05 * zeta_w * theta
    return [-grad_theta + G_theta, -grad_zeta]

def satisfies(assign, clauses, threshold=0.7):
    assign_list = list(assign)
    sat_count = 0
    for clause in clauses:
        clause_sat = False
        for lit in clause:
            var = abs(lit) - 1
            lit_val = assign_list[var] if lit > 0 else 1 - assign_list[var]
            if lit_val == 1:
                clause_sat = True
                break
        if clause_sat:
            sat_count += 1
    return sat_count / len(clauses) >= threshold

# Pipeline
start_time = time.time()

ds = build_ds(affinities, use_alt=use_alt, use_zeta=use_zeta, use_rh_gct=use_rh_gct)
eigenvals, _ = eigsh(ds, k=min(5, subsample_size), which='LM')
smoothed = hdr_smooth(eigenvals)

if use_flow:
    X0 = [0.15, 0.06]
    t_span = np.linspace(0, 1, 5)
    sol = odeint(flow_ode, X0, t_span, args=(smoothed, 1/(3*10)))
    theta_opt, zeta_opt = sol[-1]
    smoothed *= theta_opt

top_ranks = np.argsort(smoothed)[-int(0.1 * len(smoothed)):]
top_assigns = [all_assigns[i] for i in top_ranks]

gt_sols = []
samples = 1000
for _ in range(samples):
    rand_assign = tuple(np.random.randint(0, 2, n_vars))
    if satisfies(rand_assign, clauses, threshold=0.7):
        gt_sols.append(rand_assign)
gt_count = len(set(gt_sols))
sols_found = [a for a in top_assigns if satisfies(a, clauses, threshold=0.7)]
recall = len(set(sols_found)) / max(gt_count, 1)

gaps = np.diff(np.sort(smoothed))
gamma_est = max(np.min(gaps), 1e-6) / np.max(np.abs(smoothed)) if len(gaps) > 0 else 0.1  # Eps for zero gap

runtime = time.time() - start_time

print(f"RH-GCT Test (use_rh_gct={use_rh_gct}, n={n_vars}, Δ≈{m_clauses/n_vars:.1f}): γ={gamma_est:.3f}, Recall={recall:.3f}, Runtime={runtime:.2f}s, GT Sols={gt_count}")
print(f"Sols Found: {len(set(sols_found))}/{gt_count}")

# Plot
plt.figure(figsize=(8, 5))
plt.hist(smoothed, bins=8, alpha=0.7, label='RH-GCT Tuned Smoothed', color='orange')
plt.xlabel('Eigenvalues')
plt.ylabel('Density')
plt.title(f'RH-GCT Extension (γ: {gamma_est:.3f})')
plt.legend()
plt.grid(True)
plt.savefig('rh_gct_sat_cached.png')
# plt.show()  # Comment for non-interactive

print("\nTest complete! Check 'rh_gct_sat_cached.png'. For baseline: set use_rh_gct=False.")
