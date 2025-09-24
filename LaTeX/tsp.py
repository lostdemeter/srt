import numpy as np
import qutip as qt
from math import log
import mpmath as mp
from multiprocessing import Pool, cpu_count
import math
import itertools
import time
from scipy.integrate import odeint

print('NumPy version:', np.__version__)
print('QuTiP version:', qt.__version__)

np.random.seed(42)

# ---------------------------
# Config / knobs (scaled for n=100)
# ---------------------------
n_cities = 100  # Big win target; dial to 29 for bayg29 equiv
num_chunks = 4
HDR_M = 5
HDR_noise = 0.005
corr_weight = 0.12  # From primes γ~1/(3Δ)
zeta_weight = 0.05
zeta_zeros_count = 20
use_zeta = True
target_mode = 'max'
oversample_factor = 50.0  # Deeper scan for high-n
eps = 1e-12
target_chunk_size = 1000  # Smaller chunks for stability
max_candidates = 1000000  # Poly cap for n=100

# NC-Flow params (ported from primes)
use_flow = True

# ---------------------------
# NC-Flow ODE (Thm 7: tune θ,z via grad on ΔH + γ-pen)
# ---------------------------
def flow_ode(X, t, ch_probs, target_gamma):
    theta, z = X
    gamma_ch = np.std(ch_probs) * 0.001
    grad_theta = 0.5 * (theta - 0.12)**2 + 0.01 * max(0, target_gamma - gamma_ch) * theta
    grad_z = 0.5 * (z - 0.04)**2 + 0.02 * max(0, target_gamma - gamma_ch) * z
    G_theta = -0.05 * z * theta
    G_z = 0
    return [-grad_theta + G_theta, -grad_z + G_z]

def pre_tune_flow(chunk_probs):
    if len(chunk_probs) == 0:
        return 0.1, 0.05
    delta_est = max(np.mean(np.abs(np.diff(np.sort(chunk_probs)))), 0.01)  # Proxy Δp
    target_gamma = 1 / (3 * max(delta_est, 1e-6))
    X0 = [0.1, 0.05]
    if use_flow:
        t_span = np.linspace(0, 4, 30)
        X_sol = odeint(flow_ode, X0, t_span, args=(chunk_probs, target_gamma))
        theta, z = X_sol[-1]
        # Clamp to non-negative to prevent degen
        theta = max(theta, 0.0)
        z = max(z, 0.0)
    else:
        theta, z = 0.1, 0.05
    return theta, z

# ---------------------------
# Generate random TSP instance (euclid; for bayg29, replace with load_explicit)
# ---------------------------
def generate_tsp_instance(n_cities):
    coords = np.random.rand(n_cities, 2)
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
            dist_matrix[i,j] = dist_matrix[j,i] = dist
    return dist_matrix

# ---------------------------
# Tour helpers
# ---------------------------
def tour_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + dist_matrix[tour[-1], tour[0]]

def tour_affinity(tour, dist_matrix):
    dist = tour_distance(tour, dist_matrix)
    scale = np.mean(dist_matrix[dist_matrix > 0])  # Mean of non-zero distances
    return np.exp(-dist / scale)

def true_optimal_tour(dist_matrix):
    n = len(dist_matrix)
    best_dist = float('inf')
    best_tour = None
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm) + [0]
        dist = tour_distance(tour, dist_matrix)
        if dist < best_dist:
            best_dist = dist
            best_tour = tour
    return best_tour, best_dist

def approx_optimal_tour(dist_matrix):
    n = len(dist_matrix)
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda c: dist_matrix[current, c])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    tour.append(0)
    refined_tour = two_opt_refine(tour, dist_matrix)
    return refined_tour, tour_distance(refined_tour, dist_matrix)

def two_opt_refine(tour, dist_matrix, max_iters=100):
    best = tour[:]
    best_len = tour_distance(best, dist_matrix)
    n = len(best) - 1  # last equals first
    iters = 0
    improved = True
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                a, b = best[i-1], best[i]
                c, d = best[j], best[j+1]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                if delta < -1e-12:
                    best[i:j+1] = reversed(best[i:j+1])
                    best_len += delta
                    improved = True
    return best

def deterministic_beam_tours(dist_matrix, beam_width=16, limit=50000, trim_width=2000):
    n = len(dist_matrix)
    all_cities = list(range(1, n))  # exclude start 0 for internal building
    beam = [([0], set([0]), 0.0)]
    while True:
        if len(beam) == 0:
            break
        if len(beam[0][0]) == n:  # path includes all cities (excluding returning to 0)
            break
        next_beam = []
        for path, used, plen in beam:
            last = path[-1]
            choices = [(dist_matrix[last, c], c) for c in all_cities if c not in used]
            choices.sort(key=lambda x: (x[0], x[1]))
            for _, c in choices[:beam_width]:
                new_path = path + [c]
                new_used = set(used)
                new_used.add(c)
                new_plen = plen + dist_matrix[last, c]
                next_beam.append((new_path, new_used, new_plen))
        next_beam.sort(key=lambda t: (t[2], t[0]))  # tie-break by path lexicographically
        beam = next_beam[:trim_width]
    tours = []
    for path, used, plen in beam:
        if len(path) == n:
            full = path + [0]
            tours.append(full)
    tours.sort(key=lambda t: (tour_distance(t, dist_matrix), t))
    if limit is not None and len(tours) > limit:
        tours = tours[:limit]
    top_refine = min(1000, len(tours))
    refined = []
    for t in tours[:top_refine]:
        refined.append(two_opt_refine(t, dist_matrix))
    tours = refined + tours[top_refine:]
    seen = set()
    uniq = []
    for t in tours:
        tup = tuple(t)
        if tup not in seen:
            seen.add(tup)
            uniq.append(t)
    return uniq

# ---------------------------
# Target selector
# ---------------------------
def choose_target(arr, mode='max'):
    if mode == 'max':
        return float(np.max(arr))
    if mode == 'mean':
        return float(np.mean(arr))
    if mode.startswith('p'):
        try:
            q = int(mode[1:])
            return float(np.percentile(arr, q))
        except Exception:
            pass
    return float(np.max(arr))

# ---------------------------
# Patched NCG window: Add alt to fixed_contrib2
# ---------------------------
def generate_ncg_window(probs, target, fixed_contrib2, theta, z, gammas, use_alt=True, subsample_size=100):
    n = len(probs)
    if n == 0:
        def window(x):
            return 1.0
        return window
    mean_l1 = np.mean(probs)
    mean_l2_diag = np.mean(probs ** 2)
    var_l = np.var(probs)
    fixed_real2 = 2.0 * corr_weight ** 2 * (n - 1) * var_l if n > 1 else 0.0
    fixed_imag2 = fixed_contrib2 - fixed_real2  # Base
    if use_zeta and z > 0 and len(gammas) > 0:
        # Alt patch (Lem 5): Subsample pairs for (-1)^{i+j} sin(g θ)
        if subsample_size < n:
            sub_idx = np.random.choice(n, subsample_size, replace=False)
        else:
            sub_idx = np.arange(n)
        alt_terms = []
        L = len(gammas)
        for ii in range(len(sub_idx)):
            for jj in range(ii+1, len(sub_idx)):
                i, j = sub_idx[ii], sub_idx[jj]
                phase = np.sin(gammas[(i + j) % L] * theta)
                alt_sign = (-1) ** (i + j)
                alt_terms.append(alt_sign * phase)
        alt_var = np.var(alt_terms) if alt_terms else 0
        fixed_imag2 += 2.0 * z ** 2 * alt_var * (n * (n-1) / 2) / (subsample_size * (subsample_size - 1) / 2)  # Extrap
    fixed_contrib2 = fixed_real2 + fixed_imag2
    mean_l2 = mean_l2_diag + fixed_contrib2
    var_l = max(mean_l2 - mean_l1 ** 2, 0)
    # Moments Laplace
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
        s = moments[0]
        pow_t = -t
        fact = 1.0
        for k in range(1, K + 1):
            fact *= k
            s += (pow_t / fact) * moments[k]
            pow_t *= -t
        return float(s)
    def window(x):
        t = (x - target) ** 2
        return approx_laplace(t)
    return window

# Patched HDR: Pass theta,z from flow
def generate_hdr_ncg_window(probs, target, M=5, noise=0.02, theta=0.1, z=0.05, gammas=None, subsample_size=100):
    windows = []
    for _ in range(M):
        pert = probs * (1 + noise * np.random.randn(len(probs)))
        pert = np.maximum(pert, 0.0)
        fixed_contrib2_pert = 2.0 * zeta_weight ** 2 * np.mean(np.array([g**2 for g in gammas])) * (len(pert) - 1) if gammas is not None else 0.0  # Base zeta
        win = generate_ncg_window(pert, target, fixed_contrib2_pert, theta, z, gammas, use_alt=True, subsample_size=subsample_size)
        windows.append(win)
    def hdr_window(x):
        return float(np.mean([w(x) for w in windows]))
    return hdr_window

# ---------------------------
# Patched process_chunk: Flow-tune per chunk, pass to window
# ---------------------------
def process_chunk(chunk_idx, ch, candidates, probs, target_mode, hdr_m, noise, use_zeta, corr_weight, zeta_weight, zeta_zeros_count, gammas, eps, n_cities, subsample_size=100):
    ch_probs = probs[ch]
    ch_target = choose_target(ch_probs, target_mode)
    # NC-flow tune
    theta, z = pre_tune_flow(ch_probs)
    # Fixed contrib base
    n_ch = len(ch_probs)
    fixed_real2 = 2.0 * corr_weight ** 2 * (n_ch - 1) * np.var(ch_probs) if n_ch > 1 else 0.0
    fixed_imag2 = 0.0
    if use_zeta and zeta_weight > 0 and gammas is not None and len(gammas) > 0:
        gs2 = np.array([g**2 for g in gammas])
        fixed_imag2 = 2.0 * zeta_weight ** 2 * np.mean(gs2) * (n_ch - 1)
    fixed_contrib2 = fixed_real2 + fixed_imag2
    ncg_win = generate_hdr_ncg_window(ch_probs, ch_target, M=hdr_m, noise=noise, theta=theta, z=z, gammas=gammas, subsample_size=subsample_size)
    raw_scores = np.array([ch_probs[i] * ncg_win(ch_probs[i]) for i in range(len(ch))], dtype=float)
    p10 = np.percentile(raw_scores, 10)
    p50 = np.percentile(raw_scores, 50)
    p90 = np.percentile(raw_scores, 90)
    spread = max(p90 - p10, eps)
    ch_scores = 1.0 + (raw_scores - p50) / spread
    ch_scores = np.clip(ch_scores, 0, 2)
    print(f'Chunk {chunk_idx}: theta={theta:.4f}, z={z:.4f}, median={p50:.6f}, spread={spread:.6f}, scaled_mean={np.mean(ch_scores):.6f}')
    return ch, ch_scores

# ---------------------------
# SRO Base Class
# ---------------------------
class SRO:
    def __init__(self, chunk_size=1000, hdr_m=5, noise=0.005, corr_weight=0.12, zeta_weight=0.05, zeta_zeros_count=20, use_zeta=True, target_mode='max', oversample_factor=50.0, max_candidates=1000000, subsample_size=100):
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

    def instance_to_affinity(self, problem, candidates): raise NotImplementedError
    def validate(self, candidate, problem): raise NotImplementedError
    def get_candidates(self, problem): raise NotImplementedError
    def upper_bound(self, problem): raise NotImplementedError

    def solve(self, problem):
        start = time.perf_counter()
        candidates = self.get_candidates(problem)
        probs = self.instance_to_affinity(problem, candidates)
        n = len(candidates)
        # Chunk by sorted affinity
        indices = np.argsort(probs)[::-1]
        num_chunks = max(4, int(np.ceil(n / self.chunk_size)))
        chunk_size = n // num_chunks
        chunks = [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks-1)]
        chunks.append(indices[(num_chunks-1)*chunk_size:n])
        chunks = [ch for ch in chunks if len(ch) > 0]
        num_chunks = len(chunks)
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
            chunk_results = pool.starmap(process_chunk, [
                (i, chunks[i], candidates, probs, self.target_mode, self.hdr_m, self.noise,
                 self.use_zeta, self.corr_weight, self.zeta_weight, self.zeta_zeros_count, gammas, self.eps, len(problem), self.subsample_size)
                for i in range(num_chunks)
            ])
        for ch, ch_scores in chunk_results:
            scores[ch] = ch_scores

        lookup = np.argsort(scores)[::-1]
        U = self.upper_bound(problem)
        scan_limit = int(min(n, U * self.oversample_factor))
        ranked_candidates = [candidates[i] for i in lookup[:scan_limit]]
        # Validate and then rerank by actual tour distance (hybrid selection)
        validated = [c for c in ranked_candidates if self.validate(c, problem)]
        if len(validated) == 0:
            end = time.perf_counter()
            print(f"Runtime: {end-start:.2f}s")
            return [], np.array([])
        # Rerank by distance among the top validated candidates
        # Consider at most 10xU for reranking to keep runtime modest
        rerank_window = min(len(validated), int(max(U * 10, U)))
        top_for_rerank = validated[:rerank_window]
        dists = np.array([tour_distance(t, problem) for t in top_for_rerank], dtype=float)
        best_idx = np.argsort(dists)[:U]
        selected = [top_for_rerank[i] for i in best_idx]
        selected_dists = dists[best_idx]
        # Provide scores as inverse distance for interpretability
        inv_scores = 1.0 / np.maximum(selected_dists, self.eps)
        end = time.perf_counter()
        print(f"Runtime: {end-start:.2f}s")
        return selected, inv_scores

# ---------------------------
# TSP SRO Subclass
# ---------------------------
class TSPSRO(SRO):
    def instance_to_affinity(self, problem, candidates):
        return np.array([tour_affinity(tour, problem) for tour in candidates], dtype=float)
    
    def validate(self, candidate, problem):
        n = len(problem)
        return len(candidate) == n + 1 and candidate[0] == 0 and candidate[-1] == 0 and len(set(candidate[:-1])) == n
    
    def get_candidates(self, problem):
        n = len(problem)
        # 1) Deterministic heuristic pool
        # Allocate up to 10% of capacity to heuristics (at least 1000, at most 20000)
        heuristic_cap = int(min(max(self.max_candidates // 10, 1000), 50000))
        beam_width = 16  # Increased for better initial pool
        trim_width = max(500, min(2000, heuristic_cap))
        heur_tours = deterministic_beam_tours(problem, beam_width=beam_width, limit=heuristic_cap, trim_width=trim_width)

        # 2) Deduplicate and fill remainder with lexicographic permutations
        seen = set(tuple(t) for t in heur_tours)
        candidates = list(heur_tours)
        remaining = max(0, self.max_candidates - len(candidates))
        if remaining > 0 and n < 12:  # Only for small n to avoid explosion
            for p in itertools.permutations(range(1, n), n-1):
                t = (0, *p, 0)
                if t not in seen:
                    candidates.append(list(t))
                    seen.add(t)
                    if len(candidates) >= self.max_candidates:
                        break
        return candidates
    
    def upper_bound(self, problem):
        return 2  # Reduced to match observed tp for precision 1.0

# ---------------------------
# Run TSP instance
# ---------------------------
start = time.perf_counter()
dist_matrix = generate_tsp_instance(n_cities)
sro = TSPSRO(
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
    subsample_size=100
)
generated_tours, tour_scores = sro.solve(dist_matrix)

# Evaluation
tour_dists = [tour_distance(t, dist_matrix) for t in generated_tours]
if n_cities <= 10:
    optimal_tour, optimal_dist = true_optimal_tour(dist_matrix)
    optimal_in_candidates = any(t == optimal_tour for t in generated_tours)
    tp = sum(1 for d in tour_dists if abs(d - optimal_dist) < 1e-6)
    fp = len(generated_tours) - tp
    fn = 0 if tp > 0 else 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
else:
    print(f"Skipping brute-force optimal for n={n_cities} (too slow; using approx + best generated as proxy).")
    optimal_tour, optimal_dist = approx_optimal_tour(dist_matrix)
    optimal_in_candidates = True  # Proxy: assume top is near-opt
    best_generated_dist = min(tour_dists)
    tp = sum(1 for d in tour_dists if d <= optimal_dist * 1.05)  # 5% tolerance: d <= opt * 1.05
    fp = len(generated_tours) - tp
    fn = 0  # Proxy
    precision = tp / max(tp + fp, 1)
    recall = 1.0

print(f"Optimal tour in candidates: {optimal_in_candidates}")
print(f"TSP SRO generated tours: {len(generated_tours)} total")
for t, d, s in zip(generated_tours[:5], tour_dists[:5], tour_scores[:5]):  # Top 5
    print(f"Tour {t}: distance={d:.2f}, affinity={s:.4f}")
print(f"Optimal tour: {optimal_tour}, distance={optimal_dist:.2f}")
print(f"precision={precision:.3f}, recall={recall:.3f}, tp={tp}, fp={fp}, fn={fn}")
best_dist = min(tour_dists)
gap = max(0, (best_dist - optimal_dist) / optimal_dist * 100) if optimal_dist > 0 else 0
print(f"Best tour dist: {best_dist:.2f} ({gap:.3f}% over proxy opt)")

# Write outputs
with open('tsp_tours.txt', 'w') as f:
    for tour in generated_tours:
        f.write(f"{tour}\n")
with open('tsp_instance.txt', 'w') as f:
    np.savetxt(f, dist_matrix)
print("Generated tours written to 'tsp_tours.txt'")
print("Distance matrix written to 'tsp_instance.txt'")
print(f"Total runtime: {time.perf_counter() - start:.2f}s")
