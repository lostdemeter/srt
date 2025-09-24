import numpy as np
from math import log
import mpmath as mp
from multiprocessing import Pool, cpu_count
import math
import time
from scipy.integrate import odeint
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d

np.random.seed(42)

# Config
max_num = 10**9  # Test 10M; 10**8 full (10**9 requires a lot of ram, >64gb. 10**8 can be accomplished with 64gb)
HDR_M = 3  # Reduced
HDR_noise = 0.02
corr_weight = 0.12
zeta_weight = 0.05
zeta_zeros_count = 20
use_zeta = True
target_mode = 'p30'
oversample_factor = 3
eps = 1e-12
target_chunk_size = 2000  # Larger
batch_size = 100000  # Larger for less overhead
eval_mode = True
use_alt = True
use_flow = True
selection_mode = 'heap'  # one of: 'ram', 'memmap', 'heap'
memmap_path = 'scores.float32.memmap'
eval_skip_threshold = 10**9  # skip heavy eval bookkeeping beyond this n
max_scan_limit = 10_000_000  # cap the initial K to control memory/time at huge n
subsample_size = 50  # Aggressive
eig_k = 30  # Fewer modes

def prime_prob(k):
    if k < 2:
        return 0.0
    if k == 2:
        return 1.0
    return 2.0 / log(k)

def sieve_primes(n):
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    p = 2
    while p * p <= n:
        if sieve[p]:
            sieve[p*p:n+1:p] = False
        p += 1
    return np.nonzero(sieve)[0].tolist()

def miller_rabin(n):
    n = int(n)
    if n < 2:
        return False
    if n in [2, 3]:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Write n-1 = 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Fixed witnesses for deterministic test (n < 4.759e9)
    witnesses = [2, 7, 61]
    for a in witnesses:
        if a >= n:
            break
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False  # Composite
    return True  # Prime (deterministic here)

def is_prime_det(n):
    return miller_rabin(n)

def validate_batch(args):
    batch, is_prime_func = args
    return [k for k in batch if is_prime_func(k)]

def parallel_filter_primes(cands, func, batch_size=100000):
    if len(cands) == 0:
        return []
    n = len(cands)
    bs = batch_size
    num_batches = (n + bs - 1) // bs
    num_procs = min(cpu_count(), num_batches)
    batch_starts = list(range(0, n, bs))
    batch_ends = [min(start + bs, n) for start in batch_starts]
    batch_args = [(cands[start:end], func) for start, end in zip(batch_starts, batch_ends)]
    with Pool(num_procs) as pool:
        results = pool.map(validate_batch, batch_args)
    return [p for sublist in results for p in sublist]

def choose_target(arr, mode='max'):
    if mode == 'max':
        return float(np.max(arr))
    if mode == 'mean':
        return float(np.mean(arr))
    if mode.startswith('p'):
        try:
            q = int(mode[1:])
            return float(np.percentile(arr, q))
        except:
            pass
    return float(np.max(arr))

def prime_upper_bound(x):
    if x < 3:
        return x
    return int(np.ceil(x / (np.log(x) - 1.0)))

def flow_ode(X, t, ch_probs, target_gamma):
    theta, z = X
    gamma_ch = np.std(ch_probs) * 0.001
    grad_theta = (theta - 0.15)**2 + 0.1 * max(0, target_gamma - gamma_ch) * theta
    grad_z = (z - 0.06)**2 + 0.2 * max(0, target_gamma - gamma_ch) * z
    G_theta = -0.05 * z * theta
    G_z = 0
    return [-grad_theta + G_theta, -grad_z + G_z]

def pre_tune_flow(chunk_idx, ch_probs):
    delta_est = 1 / np.log(np.mean(np.arange(len(ch_probs)) + 2))
    target_gamma = (1 / (3 * delta_est)) * 0.001
    X0 = [0.1, 0.05]
    if use_flow:
        t_span = np.linspace(0, 4, 30)  # Fewer steps
        X_sol = odeint(flow_ode, X0, t_span, args=(ch_probs, target_gamma))
        theta, z = X_sol[-1]
    else:
        theta, z = 0.1, 0.05
    return theta, z

def generate_srt_window(ch_probs, theta, z, gammas, use_alt=True, subsample_size=50):
    n_ch = len(ch_probs)
    if subsample_size < n_ch:
        sub_idx = np.random.choice(n_ch, subsample_size, replace=False)
        sub_probs = ch_probs[sub_idx]
    else:
        sub_probs = ch_probs
        sub_idx = np.arange(n_ch)
    n_sub = len(sub_probs)
    # Real-valued dense matrix is sufficient; only real part is used downstream
    D_sub = np.zeros((n_sub, n_sub), dtype=np.float64)
    np.fill_diagonal(D_sub, sub_probs.astype(np.float64))
    i, j = np.triu_indices(n_sub, k=1)
    off_real = corr_weight * (sub_probs[i] - sub_probs[j])
    D_sub[i, j] = off_real
    D_sub[j, i] = off_real
    # Imaginary augmentation was discarded by taking real part; skip to reduce temporaries
    L_sub = csr_matrix(D_sub)
    eigvals_sub, _ = eigsh(L_sub, k=min(eig_k, n_sub // 2), which='LM')
    t_sub = np.median(sub_probs)
    sub_win_vec = np.vectorize(lambda x: (1 / n_sub) * np.sum(np.exp(-np.abs(eigvals_sub) * (x - t_sub)**2)))
    if subsample_size < n_ch:
        interp_x = np.sort(sub_probs)
        interp_y = sub_win_vec(interp_x)
        interp_win = interp1d(interp_x, interp_y, kind='linear', fill_value='extrapolate')
        def full_window(x):
            return interp_win(x)
        eigvals_ch = eigvals_sub
    else:
        def full_window(x):
            return sub_win_vec(x)
        eigvals_ch = eigvals_sub
    return full_window, eigvals_ch

def generate_hdr_srt_window(ch_probs, theta, z, gammas, M=3, noise=0.02, use_alt=True, subsample_size=50):
    windows = []
    eig_lists = []
    for _ in range(M):
        pert = ch_probs + noise * np.random.randn(len(ch_probs)) * np.maximum(ch_probs, eps)
        pert = np.maximum(pert, 0.0)
        win, eigs = generate_srt_window(pert, theta, z, gammas, use_alt, subsample_size)
        windows.append(win)
        eig_lists.append(eigs)
    avg_eigs = np.mean(eig_lists, axis=0)
    def hdr_window(x):
        return np.mean([w(x) for w in windows])
    hdr_window_vec = np.vectorize(hdr_window)
    return hdr_window_vec, avg_eigs

def process_chunk_score(args):
    chunk_idx, ch, theta_z = args
    theta, z = theta_z
    # Build per-chunk prime probability vector; avoid global probs if None
    if probs is not None:
        ch_probs = probs[ch]
    else:
        start = ch.start + 2
        end = ch.stop + 2
        ks = np.arange(start, end, dtype=np.float64)
        ch_probs = np.zeros(len(ks), dtype=np.float32)
        if len(ks) > 0:
            ch_probs[0] = 1.0 if start == 2 else (2.0 / np.log(ks[0])).astype(np.float32)
        if len(ks) > 1:
            ch_probs[1 if start == 2 else 0:] = (2.0 / np.log(ks[1 if start == 2 else 0:])).astype(np.float32)
    ch_target = choose_target(ch_probs, target_mode)
    ncg_win, eigvals_ch = generate_hdr_srt_window(
        ch_probs, theta, z, gammas,
        M=HDR_M, noise=HDR_noise,
        use_alt=use_alt,
        subsample_size=subsample_size
    )
    raw_scores = ch_probs * ncg_win(ch_probs)  # Vectorized!
    p10 = np.percentile(raw_scores, 10)
    p50 = np.percentile(raw_scores, 50)
    p90 = np.percentile(raw_scores, 90)
    spread = max(p90 - p10, eps)
    ch_scores = 1.0 + (raw_scores - p50) / spread
    #print(f'Chunk {chunk_idx}: theta={theta:.4f}, z={z:.4f}, median={p50:.6f}, p10={p10:.6f}, p90={p90:.6f}, scaled_mean={np.mean(ch_scores):.6f}, gamma={np.min(np.diff(np.sort(eigvals_ch))):.6f}')
    return ch, ch_scores

# Build probs / optional sieve for evaluation only at modest n
sieve_start = time.perf_counter()
if eval_mode and max_num <= eval_skip_threshold:
    true_primes = sieve_primes(max_num)
    true_count = len(true_primes)
    sieve_time = time.perf_counter() - sieve_start
    print(f"True primes <= {max_num}: {true_count} (sieve time: {sieve_time:.2f}s)")
    # Build a compact boolean mask of true primes to avoid a large Python set
    # Index i corresponds to number (i + 2)
    is_true = np.zeros(max_num - 1, dtype=bool)
    for p in true_primes:
        if p >= 2:
            is_true[p - 2] = True
else:
    true_primes = []
    true_count = 0
    is_true = None
    print(f"Skipping sieve/eval ground-truth for n={max_num} (> {eval_skip_threshold}).")
    sieve_time = time.perf_counter() - sieve_start

# Candidates are implicit [2..max_num]; avoid materializing at huge n
n = max_num - 1
# Precompute probs array only for modest n to save memory
if n <= 50_000_000:
    candidates = np.arange(2, max_num + 1, dtype=np.uint32)
    probs = np.zeros(len(candidates), dtype=np.float32)
    if len(candidates) > 0:
        probs[0] = 1.0  # P(2 is prime) = 1
    if len(candidates) > 1:
        ks = candidates[1:].astype(np.float64)
        probs[1:] = (2.0 / np.log(ks)).astype(np.float32)
else:
    candidates = None
    probs = None

# Zeta
mp.mp.dps = 30
zeros = [mp.zetazero(k) for k in range(1, zeta_zeros_count + 1)]
gammas = [float(z.imag) for z in zeros]

# Chunks
num_chunks = max(4, int(np.ceil(n / target_chunk_size)))
chunk_size = n // num_chunks
chunks = [slice(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks - 1)]
chunks.append(slice((num_chunks - 1) * chunk_size, n))

# Pre-tune
print("Pre-tuning nc-flow...")
if probs is not None:
    tuned_params = [pre_tune_flow(i, probs[chunks[i]]) for i in range(num_chunks)]
else:
    # Build per-chunk approximate probs on the fly for tuning
    tuned_params = []
    for i in range(num_chunks):
        ch = chunks[i]
        start = ch.start + 2
        end = ch.stop + 2
        ks = np.arange(start, end, dtype=np.float64)
        ch_probs = np.zeros(len(ks), dtype=np.float32)
        if len(ks) > 0:
            ch_probs[0] = 1.0 if start == 2 else (2.0 / np.log(ks[0])).astype(np.float32)
        if len(ks) > 1:
            ch_probs[1 if start == 2 else 0:] = (2.0 / np.log(ks[1 if start == 2 else 0:])).astype(np.float32)
        tuned_params.append(pre_tune_flow(i, ch_probs))

# Compute scan limit before solver so heap mode can be single-pass
U = prime_upper_bound(max_num)
scan_limit = min(n, U * oversample_factor, max_scan_limit)
print(f"Upper bound U={U}, initial scan_limit={scan_limit}")

# Solver (streaming to reduce memory)
solver_start = time.perf_counter()
num_processes = min(cpu_count(), num_chunks)

if selection_mode == 'memmap':
    # Disk-backed scores array
    scores = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(n,))
    with Pool(num_processes) as pool:
        for ch, ch_scores in pool.imap(process_chunk_score, [(i, chunks[i], tuned_params[i]) for i in range(num_chunks)]):
            scores[ch] = ch_scores
elif selection_mode == 'heap':
    import heapq
    heap = []  # (score, idx)
    with Pool(num_processes) as pool:
        for ch, ch_scores in pool.imap(process_chunk_score, [(i, chunks[i], tuned_params[i]) for i in range(num_chunks)]):
            start = ch.start
            for offset, sc in enumerate(ch_scores):
                idx = start + offset
                if len(heap) < scan_limit:
                    heapq.heappush(heap, (float(sc), idx))
                else:
                    if sc > heap[0][0]:
                        heapq.heapreplace(heap, (float(sc), idx))
else:
    # 'ram' default
    scores = np.zeros(n, dtype=np.float32)
    with Pool(num_processes) as pool:
        for ch, ch_scores in pool.imap(process_chunk_score, [(i, chunks[i], tuned_params[i]) for i in range(num_chunks)]):
            scores[ch] = ch_scores
solver_time = time.perf_counter() - solver_start
print(f"Solver time: {solver_time:.2f}s")

# Validation (unchanged)
if selection_mode == 'heap':
    # Build top-K via a min-heap without storing full scores
    # Extract indices sorted by descending score
    heap.sort(key=lambda x: x[0], reverse=True)
    lookup = np.array([idx for _, idx in heap], dtype=np.int64)
    ranked_candidates = lookup + 2  # candidates are numbers starting at 2
    # Build a cheap membership structure for the widening scan; for large n, avoid full boolean array
    if n <= 50_000_000:
        in_top = np.zeros(n, dtype=bool)
        in_top[lookup] = True
    else:
        in_top = None
elif selection_mode == 'memmap':
    scores = np.memmap(memmap_path, dtype='float32', mode='r', shape=(n,))
    top_idx = np.argpartition(scores, -scan_limit)[-scan_limit:]
    order = np.argsort(scores[top_idx])[::-1]
    lookup = top_idx[order]
    ranked_candidates = candidates[lookup]
    in_top = np.zeros(n, dtype=bool)
    in_top[lookup] = True
else:
    # 'ram'
    # Only select the top-K indices to avoid holding a full argsort of size n
    top_idx = np.argpartition(scores, -scan_limit)[-scan_limit:]
    # Order those top indices by descending score
    order = np.argsort(scores[top_idx])[::-1]
    lookup = top_idx[order]
    ranked_candidates = candidates[lookup]
    in_top = np.zeros(n, dtype=bool)
    in_top[lookup] = True

val_start = time.perf_counter()
print("Initial validation...")
filtered = parallel_filter_primes(ranked_candidates, is_prime_det, batch_size=batch_size)
gen_set = set(filtered)
tp = len(gen_set)
print(f"Initial tp={tp}/{true_count}")

if tp < true_count:
    print("Widening scan with parallel batches...")
    remaining_start = scan_limit
    # All candidates not in the initial top-K selection
    bs = batch_size
    if in_top is None:
        # Large-n: stream ranges [2..max_num] excluding top indices
        top_set = set(int(i) for i in lookup.tolist())
        batch_args = []
        for start in range(2, max_num + 1, bs):
            end = min(start + bs, max_num + 1)
            # Build batch list excluding top-K by index (k -> idx=k-2)
            batch = [k for k in range(start, end) if (k - 2) not in top_set]
            if batch:
                batch_args.append((batch, is_prime_det))
    else:
        # Modest n: we have candidates and in_top mask
        remaining_cands = candidates[~in_top]
        remaining_n = len(remaining_cands)
        batch_starts = list(range(0, remaining_n, bs))
        batch_ends = [min(s + bs, remaining_n) for s in batch_starts]
        batch_args = [(remaining_cands[s:e], is_prime_det) for s,e in zip(batch_starts, batch_ends)]
    if not batch_args:
        print("No remaining candidates to scan.")
    else:
        num_procs_val = max(1, min(cpu_count(), len(batch_args)))
        with Pool(num_procs_val) as val_pool:
            for batch_primes in val_pool.imap(validate_batch, batch_args):
                filtered.extend(batch_primes)
                gen_set.update(batch_primes)
                tp = len(gen_set)
                print(f"Widened batch: added {len(batch_primes)} primes, total tp={tp}/{true_count}")
                if tp >= true_count:
                    break
if tp < true_count:
    print(f"Warning: Still missing {true_count - tp} primes after full scan.")

val_time = time.perf_counter() - val_start
print(f"Validation time: {val_time:.2f}s")

gen_time = sieve_time + solver_time + val_time
print(f"Total gen time: {gen_time:.2f}s")

generated_primes = sorted(filtered)

with open('generated_primes.txt', 'w') as f:
    for p in generated_primes:
        f.write(f"{p}\n")
print("Generated primes written to 'generated_primes.txt'")

if eval_mode and n <= eval_skip_threshold:
    eval_start = time.perf_counter()
    # Build a compact boolean mask for generated primes
    is_gen = np.zeros(n, dtype=bool)
    for p in generated_primes:
        is_gen[p - 2] = True
    tp = int(np.count_nonzero(is_gen & is_true))
    fp = int(np.count_nonzero(is_gen & ~is_true))
    fn = int(np.count_nonzero(~is_gen & is_true))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    # Build a compact inverse-rank array for only the top scan_limit candidates to avoid
    # constructing a huge Python dict. This keeps memory bounded (~4 bytes * n).
    inv_rank = np.full(n, -1, dtype=np.int32)
    inv_rank[lookup[:scan_limit]] = np.arange(min(scan_limit, n), dtype=np.int32)
    prime_ranks = [int(inv_rank[p - 2]) for p in true_primes]
    max_witness_rank = max(prime_ranks) if prime_ranks and max(prime_ranks) >= 0 else 0
    eval_time = time.perf_counter() - eval_start
    print(f"Solver time: {solver_time:.2f}s")
    print(f'Generated primes: {len(generated_primes)} total')
    print(f'precision={precision:.3f}, recall={recall:.3f}, tp={tp}, fp={fp}, fn={fn}')
    print(f'upper_bound={U}, true_count={true_count}')
    print(f'Max witness rank: {max_witness_rank} (O(n log n) ~{int(n * log(n)/log(2))})')
    print(f"Eval time: {eval_time:.2f}s")
    if fn > 0:
        missing_idx = np.where(is_true & ~is_gen)[0][:10]
        missing = [int(i + 2) for i in missing_idx]
        print('Missing primes:', missing[:10], '...')
else:
    print("Skipping eval.")
