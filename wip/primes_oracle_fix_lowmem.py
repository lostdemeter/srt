import numpy as np
from math import log
import mpmath as mp
import time
import argparse
from scipy.integrate import odeint
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count  # Optional; fallback single-core

np.random.seed(42)

# Config (low-mem tweaks)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10**6, help='Max n for primes')
    parser.add_argument('--full_srt', action='store_true', help='Use full D_S/eig/flow')
    return parser.parse_args()

args = get_args()
max_num = args.n
HDR_M = 3
HDR_noise = 0.02
corr_weight = 0.12
zeta_weight = 0.05
zeta_zeros_count = 20
L = 20  # Zeta mod L
use_zeta = True
target_mode = 'p30'
oversample_factor = 3
eps = 1e-12
target_chunk_size = 1000  # Smaller chunks
batch_size = 10000  # Small batches
eval_mode = True
use_alt = True
use_flow = True
selection_mode = 'memmap'  # Low-mem
memmap_path = 'scores.memmap'
eval_skip_threshold = 10**8  # Skip heavy for >10^8
subsample_size = 50
eig_k = 10  # Fewer modes
use_full_srt = args.full_srt

print(f"Low-mem primes oracle for n={max_num} (full SRT: {use_full_srt})")

def prime_prob(k):
    if k < 2: return 0.0
    if k == 2: return 1.0
    return 2.0 / log(k)

def miller_rabin(n):
    n = int(n)
    if n < 2: return False
    if n in [2, 3]: return True
    if n % 2 == 0 or n % 3 == 0: return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    witnesses = [2, 7, 61]
    for a in witnesses:
        if a >= n: break
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True

def is_prime_det(n):
    return miller_rabin(n)

def validate_batch(args):
    batch, is_prime_func = args
    return [k for k in batch if is_prime_func(k)]

def parallel_filter_primes(cands, func, batch_size=10000, num_procs=1):  # Single-core fallback
    if len(cands) == 0: return []
    n = len(cands)
    bs = batch_size
    num_batches = (n + bs - 1) // bs
    num_procs = min(num_procs, num_batches)  # Low-mem: 1 proc
    batch_starts = list(range(0, n, bs))
    batch_ends = [min(start + bs, n) for start in batch_starts]
    batch_args = [(cands[start:end], func) for start, end in zip(batch_starts, batch_ends)]
    if num_procs > 1:
        with Pool(num_procs) as pool:
            results = pool.map(validate_batch, batch_args)
    else:
        results = [validate_batch(arg) for arg in batch_args]
    return [p for sublist in results for p in sublist]

# Memmap scores (low-mem)
scores = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(max_num - 1,))
for i in range(2, max_num + 1):
    scores[i-2] = prime_prob(i)

# Zeta zeros pre-cache
zeta_zeros = [float(mp.zetazero(k).imag) for k in range(1, zeta_zeros_count + 1)]

def build_ds_subsample(aff_sub, subsample_size):
    n_sub = len(aff_sub)
    data = aff_sub.copy()
    rows = np.arange(n_sub, dtype=int)
    cols = rows.copy()
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            gamma_ij = corr_weight * np.random.rand()
            phase = zeta_zeros[(i + j) % L] if use_zeta else np.random.rand()
            delta = ((-1)**(i + j) * gamma_ij * phase if use_alt else gamma_ij * phase)
            data = np.append(data, [delta, np.conj(delta)])
            rows = np.append(rows, [i, j])
            cols = np.append(cols, [j, i])
    ds_sparse = csr_matrix((data, (rows, cols)), shape=(n_sub, n_sub))
    return ds_sparse

def hdr_smooth(eigenvals, M=HDR_M, sigma0=0.1):
    smoothed = eigenvals.real.copy()
    for m in range(M):
        eps_m = np.random.uniform(-HDR_noise, HDR_noise)
        sigma_m = sigma0 + eps_m
        phi_m = np.random.uniform(0, 2 * np.pi)
        t = np.mean(smoothed)
        damp = np.exp(-sigma_m**2 * (smoothed - t)**2 + 1j * phi_m * smoothed).real / M
        smoothed += damp
    return smoothed

def flow_ode(X, t, smoothed_evals, target_gamma=1/(3*1)):  # Δ=1 for primes
    theta, zeta_w = X
    L_loss = np.mean((smoothed_evals - np.mean(smoothed_evals))**2) + zeta_weight * np.var(smoothed_evals)
    grad_theta = 2 * (theta - 0.15) + 0.1 * max(0, target_gamma - np.std(smoothed_evals)) * theta
    grad_zeta = 2 * (zeta_w - 0.06) + 0.2 * max(0, target_gamma - np.std(smoothed_evals)) * zeta_w
    G_theta = -0.05 * zeta_w * theta
    return [-grad_theta + G_theta, -grad_zeta]

# Incremental low-mem pipeline
primes = []
prev_n = 0
start_time = time.time()

for chunk_start in range(0, max_num - 1, target_chunk_size):
    chunk_end = min(chunk_start + target_chunk_size, max_num - 1)
    chunk_cands = range(prev_n + 2, chunk_end + 2)
    chunk_scores = scores[prev_n:chunk_end]
    
    if use_full_srt:
        # Subsample for D_S (low-mem)
        sub_indices = np.random.choice(len(chunk_scores), min(subsample_size, len(chunk_scores)), replace=False)
        aff_sub = chunk_scores[sub_indices]
        ds_sub = build_ds_subsample(aff_sub, len(aff_sub))
        eigenvals, eigenvecs = eigsh(ds_sub, k=min(eig_k, len(aff_sub)), which='LM')  # Get evecs
        smoothed = hdr_smooth(eigenvals)
        if use_flow:
            X0 = [0.15, 0.06]
            t_span = np.linspace(0, 1, 5)
            sol = odeint(flow_ode, X0, t_span, args=(smoothed, 1/(3*1)))
            theta_opt, _ = sol[-1]
            smoothed *= theta_opt
        # Fixed: Project aff_sub onto top evecs for full n_sub ranks (len = subsample_size)
        projections = np.dot(aff_sub, eigenvecs[:, :min(3, eig_k)])  # Project to top 3 evecs
        sub_ranks = np.sum(projections**2, axis=1)  # Squared norms for resonance score
        # Interp sub_ranks to full chunk_scores
        interp_func = interp1d(sub_indices, sub_ranks, kind='linear', fill_value='extrapolate')
        chunk_smoothed = interp_func(np.arange(len(chunk_scores)))
        top_ranks = np.argsort(chunk_smoothed)[-int(0.1 * len(chunk_smoothed)):]
        candidates = np.array(chunk_cands)[top_ranks]
    else:
        candidates = np.array(chunk_cands)
    
    chunk_primes = parallel_filter_primes(candidates.tolist(), is_prime_det)
    primes.extend(chunk_primes)
    prev_n = chunk_end

elapsed = time.time() - start_time
num_primes = len(primes)
print(f"n={max_num}: {num_primes} primes in {elapsed:.2f}s (low-mem mode)")

# Cleanup
del scores

# Optional: Compare to sieve for precision/recall (low-mem sieve for small n)
if max_num <= 10**5:
    sieve_primes_list = [k for k in range(2, max_num + 1) if is_prime_det(k)]
    precision = len(set(primes) & set(sieve_primes_list)) / len(primes) if primes else 1.0
    recall = len(set(primes) & set(sieve_primes_list)) / len(sieve_primes_list) if sieve_primes_list else 1.0
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
else:
    print("Precision/Recall: Skip for large n (use Miller-Rabin det)")

print("Low-mem oracle complete! Delete scores.memmap if done.")
