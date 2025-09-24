import numpy as np
import qutip as qt
from math import log
import mpmath as mp
from multiprocessing import Pool, cpu_count
import math
import itertools
import time
from scipy.integrate import odeint
import random

print('NumPy version:', np.__version__)
print('QuTiP version:', qt.__version__)

np.random.seed(42)

# ---------------------------
# Config / knobs
# ---------------------------
n_vars = 40  # Test first; =40 for wow
m_clauses = 160  # =160 for 40
num_chunks = 4
HDR_M = 5
HDR_noise = 0.02
corr_weight = 0.12  # From TSP γ~1/(3Δ)
zeta_weight = 0.05
zeta_zeros_count = 20
use_zeta = True
target_mode = 'p30'
oversample_factor = 10.0  # Increased to scan deeper for full recall
eps = 1e-12
target_chunk_size = 5000  # Larger chunks for efficiency on 1M candidates
max_candidates = 1048576 if n_vars <= 20 else 50000  # Full enum small, beam large
use_flow = True

# ---------------------------
# Soft ODE Flow
# ---------------------------
def flow_ode(X, t, ch_probs, target_gamma):
    theta, z = X
    gamma_ch = np.std(ch_probs) * 0.001
    grad_theta = 0.5 * (theta - 0.12) ** 2 + 0.01 * max(0, target_gamma - gamma_ch) * theta
    grad_z = 0.5 * (z - 0.04) ** 2 + 0.02 * max(0, target_gamma - gamma_ch) * z
    G_theta = -0.05 * z * theta
    G_z = 0
    return [-grad_theta + G_theta, -grad_z + G_z]

def pre_tune_flow(chunk_probs):
    if len(chunk_probs) == 0:
        return 0.1, 0.05
    delta_est = max(np.mean(np.abs(np.diff(np.sort(chunk_probs)))), 0.01)
    target_gamma = 1 / (3 * delta_est)
    X0 = [0.1, 0.05]
    if use_flow:
        t_span = np.linspace(0, 4, 30)
        X_sol = odeint(flow_ode, X0, t_span, args=(chunk_probs, target_gamma))
        theta, z = X_sol[-1]
        theta = max(theta, 0.0)
        z = max(z, 0.0)
    else:
        theta, z = 0.1, 0.05
    return theta, z

# ---------------------------
# Helper: Generate random 3-SAT instance
# ---------------------------
def generate_3sat(n_vars, m_clauses):
    clauses = []
    for _ in range(m_clauses):
        vars = np.random.choice(range(1, n_vars + 1), size=3, replace=False)
        literals = [v if np.random.rand() > 0.5 else -v for v in vars]
        clauses.append(literals)
    return clauses

# ---------------------------
# Optimized Affinity: Precompute shared matrix
# ---------------------------
def precompute_shared_matrix(clauses):
    m = len(clauses)
    shared = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(i + 1, m):
            set_i = set(abs(lit) for lit in clauses[i])
            set_j = set(abs(lit) for lit in clauses[j])
            shared[i, j] = shared[j, i] = len(set_i & set_j)
    return shared

def sat_affinity_vectorized(assignments, clauses, shared_matrix, var_freq, weights):
    """Compute an affinity score per assignment with conflict penalties.

    Affinity = normalized weighted satisfaction - penalty
    where penalty increases when an assignment satisfies clause i while
    leaving many strongly-related clauses j unsatisfied.

    The result is min-max normalized to [0, 1] across the batch to ensure
    usable spread for ranking.
    """
    m = len(clauses)
    n_ass = len(assignments)

    # Clause satisfaction matrix: (n_ass x m)
    satisfied = np.zeros((n_ass, m), dtype=bool)
    for i, clause in enumerate(clauses):
        clause_sat = np.zeros(n_ass, dtype=bool)
        for lit in clause:
            sign = 1 if lit > 0 else 0
            clause_sat |= (assignments[:, abs(lit) - 1] == sign)
        satisfied[:, i] = clause_sat

    # Weighted satisfaction normalized to [0,1]
    w = np.asarray(weights, dtype=np.float64) if weights is not None and len(weights) == m else np.ones(m, dtype=np.float64)
    w_sum = max(np.sum(w), 1e-9)
    norm_sat = (satisfied.astype(np.float64) * w).sum(axis=1) / w_sum  # shape (n_ass,)

    # Conflict penalty using shared_matrix
    if shared_matrix is not None and getattr(shared_matrix, 'size', 0) > 0:
        W = np.asarray(shared_matrix, dtype=np.float64)
        max_shared = max(np.max(W), 1.0)
        W = W / max_shared
        np.fill_diagonal(W, 0.0)
    else:
        W = np.zeros((m, m), dtype=np.float64)

    P = satisfied.astype(np.float64)  # (n x m)
    NotP = 1.0 - P                    # (n x m)
    alpha = 0.05  # Milder penalty
    penalties = alpha * np.sum(P * (NotP @ W.T), axis=1)

    raw = norm_sat - penalties
    raw_min = float(np.min(raw))
    raw_max = float(np.max(raw))
    if raw_max > raw_min + 1e-12:
        affinities = (raw - raw_min) / (raw_max - raw_min)
    else:
        affinities = np.zeros_like(raw)
    return affinities

# ---------------------------
# Helper: Check if assignment satisfies clause
# ---------------------------
def clause_satisfied(assignment, clause):
    return any(assignment[abs(lit) - 1] == (1 if lit > 0 else 0) for lit in clause)

# ---------------------------
# Target selector
# ---------------------------
def choose_target(arr, mode='max'):
    if mode == 'max':
        return float(np.max(arr))
    if mode == 'mean':
        return float(np.mean(arr))
    if isinstance(mode, str) and mode.startswith('p'):
        try:
            q = int(mode[1:])
            return float(np.percentile(arr, q))
        except Exception:
            pass
    return float(np.max(arr))

# ---------------------------
# O(n) HDR NCG window using moments
# ---------------------------
def generate_ncg_window(probs, target, fixed_contrib2, theta, z, gammas, subsample_size=100):
    n = len(probs)
    if n == 0:
        def window(x):
            return 1.0
        return window
    mean_l1 = np.mean(probs)
    mean_l2_diag = np.mean(probs ** 2)
    var_l = np.var(probs)
    fixed_real2 = 2.0 * corr_weight ** 2 * (n - 1) * var_l if n > 1 else 0.0
    fixed_imag2 = fixed_contrib2 - fixed_real2
    if use_zeta and z > 0 and len(gammas) > 0:
        if subsample_size < n:
            sub_idx = np.random.choice(n, subsample_size, replace=False)
        else:
            sub_idx = np.arange(n)
        alt_terms = []
        L = len(gammas)
        for ii in range(len(sub_idx)):
            for jj in range(ii + 1, len(sub_idx)):
                i, j = sub_idx[ii], sub_idx[jj]
                phase = np.sin(gammas[(i + j) % L] * theta)
                alt_sign = (-1) ** (i + j)
                alt_terms.append(alt_sign * phase)
        alt_var = np.var(alt_terms) if alt_terms else 0
        fixed_imag2 += 2.0 * z ** 2 * alt_var * (n * (n - 1) / 2) / (subsample_size * (subsample_size - 1) / 2)
    fixed_contrib2 = fixed_real2 + fixed_imag2
    mean_l2 = mean_l2_diag + fixed_contrib2
    var_l = max(mean_l2 - mean_l1 ** 2, 0)
    K = 4
    moments = np.zeros(K + 1)
    moments[0] = 1.0
    moments[1] = mean_l1
    moments[2] = mean_l2
    moments[3] = mean_l1 ** 3 + 3 * mean_l1 * var_l
    moments[4] = mean_l1 ** 4 + 6 * mean_l1 ** 2 * var_l + 3 * var_l ** 2

    def approx_laplace(t):
        if t < 1e-10:
            return 1.0
        s = 0.0
        s += moments[0]
        pow_t = -t
        fact = 1.0
        for k in range(1, K + 1):
            fact *= k
            s += (pow_t / fact) * moments[k]
            pow_t *= -t
        return float(s)

    def window(x):
        dx2 = (x - target) * (x - target)
        return approx_laplace(dx2)

    return window

def generate_hdr_ncg_window(probs, target, fixed_imag2, corr_weight, M=5, noise=0.02, eps=1e-12, theta=0.1, z=0.05, gammas=None, subsample_size=100):
    n = len(probs)
    windows = []
    for _ in range(M):
        pert = np.maximum(probs * (1 + noise * np.random.randn(n)), eps)
        mean_l1 = float(np.mean(pert))
        mean_l2_diag = float(np.mean(pert ** 2))
        var_pert = float(np.var(pert))
        fixed_real2 = 2.0 * corr_weight ** 2 * (n - 1) * var_pert if n > 1 else 0.0
        fixed_contrib2_pert = fixed_real2 + fixed_imag2
        win = generate_ncg_window(pert, target, fixed_contrib2_pert, theta, z, gammas, subsample_size=subsample_size)
        windows.append(win)

    def hdr_window(x):
        return float(np.mean([w(x) for w in windows]))

    return hdr_window

# ---------------------------
# Process chunk
# ---------------------------
def process_chunk(chunk_idx, ch, candidates, probs, target_mode, HDR_M, HDR_noise, use_zeta, corr_weight, zeta_weight, gammas, eps, subsample_size):
    ch_probs = probs[ch]
    if not np.all(np.isfinite(ch_probs)) or np.any(ch_probs < 0):
        print(f"Warning: Invalid probs in chunk {chunk_idx}: {ch_probs}")
        ch_probs = np.clip(ch_probs, 0, 1)
    ch_target = choose_target(ch_probs, target_mode)
    theta, z = pre_tune_flow(ch_probs)
    n_ch = len(ch_probs)
    fixed_imag2 = 0.0
    if use_zeta and zeta_weight > 0 and len(gammas) > 0 and n_ch > 1:
        L = len(gammas)
        freq = np.bincount(np.arange(n_ch) % L, minlength=L)
        gs2 = np.array([float(g) ** 2 for g in gammas])
        sum_imag2_ij = 0.0
        for s in range(L):
            num_pairs = 0.0
            for a in range(L):
                b = (s - a) % L
                fab = freq[a]
                fbb = freq[b]
                if a < b:
                    num_pairs += fab * fbb
                elif a == b:
                    num_pairs += fab * (fab - 1) / 2
            sum_imag2_ij += num_pairs * gs2[s]
        fixed_imag2 = 2.0 * zeta_weight ** 2 * sum_imag2_ij / n_ch
        if z > 0 and use_zeta and len(gammas) > 0:
            if subsample_size < n_ch:
                sub_idx = np.random.choice(n_ch, subsample_size, replace=False)
            else:
                sub_idx = np.arange(n_ch)
            alt_terms = []
            L = len(gammas)
            for ii in range(len(sub_idx)):
                for jj in range(ii + 1, len(sub_idx)):
                    i, j = sub_idx[ii], sub_idx[jj]
                    phase = np.sin(gammas[(i + j) % L] * theta)
                    alt_sign = (-1) ** (i + j)
                    alt_terms.append(alt_sign * phase)
            alt_var = np.var(alt_terms) if len(alt_terms) > 1 else 0
            extrap = (n_ch * (n_ch - 1) / 2) / (len(sub_idx) * (len(sub_idx) - 1) / 2) if len(sub_idx) > 1 else 1
            fixed_imag2 += 2.0 * z ** 2 * alt_var * extrap
    ncg_win = generate_hdr_ncg_window(
        ch_probs, ch_target, fixed_imag2, corr_weight,
        M=HDR_M, noise=HDR_noise, eps=eps, theta=theta, z=z, gammas=gammas, subsample_size=subsample_size
    )
    raw_scores = np.array([ch_probs[i] * ncg_win(ch_probs[i]) for i in range(n_ch)], dtype=float)
    if not np.all(np.isfinite(raw_scores)):
        print(f"Warning: Invalid raw_scores in chunk {chunk_idx}: {raw_scores}")
        raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
    p10 = np.percentile(raw_scores, 10)
    p50 = np.percentile(raw_scores, 50)
    p90 = np.percentile(raw_scores, 90)
    spread = max(p90 - p10, eps * 10)
    ch_scores = 1.0 + (raw_scores - p50) / spread
    ch_scores = np.clip(ch_scores, 0, 2)
    if spread < eps * 10:
        print(f"Warning: Low spread chunk {chunk_idx}: {spread}")
    print(f'Chunk {chunk_idx}: theta={theta:.4f}, z={z:.4f}, median={p50:.6f}, p10={p10:.6f}, p90={p90:.6f}, scaled_mean={np.mean(ch_scores):.6f}')
    return ch, ch_scores

# ---------------------------
# Beam candidates (WalkSAT seed + deep greedy + guided mutate)
# ---------------------------
def mutate_assignment(ass, n_vars, num_flips=5):
    mut = ass[:]
    flips = random.sample(range(n_vars), min(num_flips, n_vars))
    for f in flips:
        mut[f] = 1 - mut[f]
    return mut


def get_beam_candidates(clauses, n_vars, beam_size=5000, mutate_count=40000, depth=15):
    # Boosted WalkSAT-like seed from random starts
    def walk_sat(clauses, n_vars, max_iters=200, starts=500):
        sols = []
        for _ in range(starts):
            ass = [random.randint(0, 1) for _ in range(n_vars)]
            violated = None
            for _ in range(max_iters):
                violated = [c for c in clauses if not clause_satisfied(ass, c)]
                if not violated:
                    sols.append(tuple(ass))
                    break
                c = random.choice(violated)
                lit = random.choice(c)
                v = abs(lit) - 1
                flip_val = (1 if lit > 0 else 0)
                ass[v] = flip_val
            if violated is not None and not violated:
                sols.append(tuple(ass))
        # Dedup
        sols = list(dict.fromkeys(sols))
        return [list(s) for s in sols]

    seeds = walk_sat(clauses, n_vars)
    print(f"Seeds from WalkSAT: {len(seeds)}")
    partials = list(seeds)

    # Var order: frequency descending (most constrained first)
    var_freq = np.zeros(n_vars, dtype=int)
    for c in clauses:
        for lit in c:
            var_freq[abs(lit) - 1] += 1
    var_order = np.argsort(-var_freq)

    def greedy_unit_prop(assignment, pos):
        if pos >= n_vars:
            return [assignment[:]]
        v = var_order[pos]
        sols = []
        for val in (0, 1, random.randint(0, 1)):
            temp = assignment[:]
            temp[v] = val
            # prune only if ALL clauses are fully falsified (very loose)
            fully_unsat = all(
                all((temp[abs(lit) - 1] == (0 if lit > 0 else 1)) for lit in c)
                for c in clauses
            )
            if fully_unsat:
                continue
            if pos + 1 >= depth:
                fill = temp[:]
                for r in range(n_vars):
                    if fill[r] is None:
                        fill[r] = random.randint(0, 1)
                sols.append(fill)
            else:
                sols.extend(greedy_unit_prop(temp, pos + 1))
            if len(sols) >= beam_size // 2:
                break
        return sols[: beam_size // 2]

    if partials:
        greedy_partials = greedy_unit_prop(random.choice(partials), 0)
    else:
        greedy_partials = greedy_unit_prop([None] * n_vars, 0)
    print(f"Partials from greedy: {len(greedy_partials)}")
    partials.extend(greedy_partials)

    if len(partials) < beam_size:
        partials.extend([[random.randint(0, 1) for _ in range(n_vars)] for _ in range(beam_size - len(partials))])

    candidates = partials[:beam_size]
    seen = set(tuple(c) for c in candidates)

    # Guided mutate to grow to max_candidates
    for _ in range(mutate_count):
        base = random.choice(partials)
        violated = [c for c in clauses if not clause_satisfied(base, c)]
        if violated:
            c = random.choice(violated)
            lit = random.choice(c)
            v = abs(lit) - 1
            flip_val = (1 if lit > 0 else 0)
            mut = base[:]
            mut[v] = flip_val
            # Chain flip 1-2 more guided steps
            for __ in range(random.randint(1, 2)):
                violated2 = [cc for cc in clauses if not clause_satisfied(mut, cc)]
                if not violated2:
                    break
                cc = random.choice(violated2)
                lit2 = random.choice(cc)
                v2 = abs(lit2) - 1
                mut[v2] = (1 if lit2 > 0 else 0)
        else:
            mut = mutate_assignment(base, n_vars, num_flips=random.randint(5, 8))
        tpl = tuple(mut)
        if tpl not in seen:
            candidates.append(mut)
            seen.add(tpl)
            if len(candidates) >= max_candidates:
                break

    return candidates[:max_candidates]

# ---------------------------
# SRO Base Class
# ---------------------------
class SRO:
    def __init__(self, chunk_size=5000, hdr_m=5, noise=0.02, corr_weight=0.05, zeta_weight=0.05, zeta_zeros_count=20, use_zeta=True, target_mode='p30', oversample_factor=10.0, max_candidates=1048576, subsample_size=100):
        self.chunk_size = chunk_size
        self.hdr_m = hdr_m
        self.noise = noise
        self.corr_weight = corr_weight
        self.zeta_weight = zeta_weight
        self.zeta_zeros_count = zeta_zeros_count
        self.use_zeta = use_zeta
        self.target_mode = target_mode
        self.oversample_factor = oversample_factor
        self.max_candidates = max_candidates
        self.subsample_size = subsample_size
        self.eps = 1e-12

    def instance_to_affinity(self, problem, candidates, shared_matrix, var_freq, weights):
        raise NotImplementedError

    def validate(self, candidate, problem):
        raise NotImplementedError

    def get_candidates(self, problem):
        raise NotImplementedError

    def upper_bound(self, problem):
        return 100000  # For n=40 ~1e5

    def solve(self, problem, shared_matrix, var_freq, weights):
        start = time.perf_counter()
        candidates = self.get_candidates(problem)
        print(f"Generated {len(candidates)} candidates")
        affinity_start = time.perf_counter()
        probs = self.instance_to_affinity(problem, candidates, shared_matrix, var_freq, weights)
        affinity_time = time.perf_counter() - affinity_start
        print(f"Affinity computation time: {affinity_time:.2f}s")
        n = len(candidates)
        num_chunks = max(4, int(np.ceil(n / self.chunk_size)))
        chunk_size = n // num_chunks
        chunks = [slice(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks - 1)]
        chunks.append(slice((num_chunks - 1) * chunk_size, n))
        scores = np.zeros(n, dtype=float)

        if self.use_zeta and self.zeta_weight > 0 and self.zeta_zeros_count > 0:
            mp.mp.dps = 30
            zeros = [mp.zetazero(k) for k in range(1, self.zeta_zeros_count + 1)]
            gammas = [float(z.imag) for z in zeros]
        else:
            gammas = []

        print(f"Affinity distribution: {np.histogram(probs, bins=20)[1]}")
        num_processes = min(cpu_count(), num_chunks)
        with Pool(num_processes) as pool:
            chunk_results = pool.starmap(
                process_chunk,
                [
                    (
                        i,
                        chunks[i],
                        candidates,
                        probs,
                        self.target_mode,
                        self.hdr_m,
                        self.noise,
                        self.use_zeta,
                        self.corr_weight,
                        self.zeta_weight,
                        gammas,
                        self.eps,
                        self.subsample_size,
                    )
                    for i in range(num_chunks)
                ],
            )
        for ch, ch_scores in chunk_results:
            scores[ch] = ch_scores

        lookup = np.argsort(scores)[::-1]
        U = self.upper_bound(problem)
        scan_limit = int(min(n, U * self.oversample_factor))
        top_indices = lookup[:scan_limit]
        ranked_candidates = [candidates[int(i)] for i in top_indices]
        filtered = [c for c in ranked_candidates if self.validate(c, problem)]
        if len(filtered) < U:
            scanned = scan_limit
            for j, idx in enumerate(lookup[scan_limit:]):
                c = candidates[int(idx)]
                scanned += 1
                if self.validate(c, problem):
                    filtered.append(c)
                if len(filtered) >= U:
                    break
            print(f"Widened scan to {scanned} candidates to find {len(filtered)} solutions")
        end = time.perf_counter()
        print(f"SRO Runtime (post-candidates): {end-start:.2f}s")
        return sorted(filtered, key=lambda x: sum(x))

# ---------------------------
# 3-SAT SRO Subclass
# ---------------------------
class SAT3SRO(SRO):
    def instance_to_affinity(self, problem, candidates, shared_matrix, var_freq, weights):
        return sat_affinity_vectorized(np.array(candidates), problem, shared_matrix, var_freq, weights)

    def validate(self, candidate, problem):
        return all(clause_satisfied(candidate, c) for c in problem)

    def get_candidates(self, problem):
        n_vars_local = max(max(abs(lit) for lit in clause) for clause in problem)
        if n_vars_local <= 20:
            all_cands = [list(ass) for ass in itertools.product([0, 1], repeat=n_vars_local)]
            return all_cands[:max_candidates]
        else:
            return get_beam_candidates(problem, n_vars_local, beam_size=5000, mutate_count=40000, depth=15)

    def upper_bound(self, problem):
        return super().upper_bound(problem)

# ---------------------------
# Run multiple instances (solver only)
# ---------------------------
for i in range(3):
    print(f"\nRunning instance {i+1}")
    clauses = generate_3sat(n_vars, m_clauses)
    shared_matrix = precompute_shared_matrix(clauses)
    var_freq = np.zeros(n_vars)
    for c in clauses:
        for lit in c:
            var_freq[abs(lit) - 1] += 1
    weights = [1.0 / (1 + sum(var_freq[abs(lit) - 1] for lit in c)) for c in clauses]
    sro = SAT3SRO(
        chunk_size=target_chunk_size,
        hdr_m=HDR_M,
        noise=HDR_noise,
        corr_weight=corr_weight,
        zeta_weight=zeta_weight,
        zeta_zeros_count=zeta_zeros_count,
        use_zeta=use_zeta,
        target_mode=target_mode,
        oversample_factor=oversample_factor,
        max_candidates=max_candidates,
        subsample_size=100,
    )
    generated_solutions = sro.solve(clauses, shared_matrix, var_freq, weights)

    # Verifier
    n_vars_local = max(max(abs(lit) for lit in clause) for clause in clauses)
    if n_vars_local <= 25:
        total_sols = [ass for ass in itertools.product([0, 1], repeat=n_vars_local) if sro.validate(ass, clauses)]
        recall = len(generated_solutions) / len(total_sols) if total_sols else 0.0
        print(f"Instance {i+1} - Exact recall: {recall:.4f}")
        print("Instance {i+1} - Precision: 1.0000 (validated)")
    else:
        sample_size = 100000
        found_set = set(tuple(s) for s in generated_solutions)
        tp = 0
        sample_sols_count = 0
        for ass in itertools.islice(itertools.product([0, 1], repeat=n_vars_local), sample_size):
            if sro.validate(ass, clauses):
                sample_sols_count += 1
                if tuple(ass) in found_set:
                    tp += 1
        recall_est = tp / sample_sols_count if sample_sols_count > 0 else 0.0
        from math import sqrt, log
        n_sample = sample_size if sample_sols_count > 0 else 1
        ci = sqrt(log(20) / (2 * n_sample))
        print(f"Instance {i+1} - Estimated recall: {recall_est:.4f} ±{ci:.4f}")
        print("Instance {i+1} - Precision: 1.0000 (validated)")

    print(f"Instance {i+1} - 3-SAT SRO generated solutions: {len(generated_solutions)}")

    with open(f'3sat_solutions_{i}.txt', 'w') as f:
        for sol in generated_solutions:
            f.write(f"{list(sol)}\n")
    with open(f'3sat_instance_{i}.txt', 'w') as f:
        for c in clauses:
            f.write(f"{c}\n")
    print(f"Instance {i+1} solutions written to '3sat_solutions_{i}.txt'")
    print(f"Instance {i+1} 3-SAT instance written to '3sat_instance_{i}.txt'")
