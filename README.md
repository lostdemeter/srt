# Spectral Resonance Theory (SRT): Implementations for Emergent Harmony and Polynomial Optimization

This repository provides Python implementations of **Spectral Resonance Theory (SRT)**, a noncommutative geometry (NCG) framework for harmonic synchronization across discrete-continuous boundaries. SRT recasts optimization as eigenvalue overlaps in deformed spectral triples, using axioms for Hermitian Dirac operators from affinity states to generate non-boxcar windows via High Dynamic Range (HDR) ensembles. It proves resonance principles, boundary preservation (e.g., Borwein integrals to L=20), and scalability for nested sets, with lemmas extending to ∆-regular NP problems for poly-time witnesses (O(n log n) ranks), alternating stabilization, nc-spectral flow, and SDP relaxations ensuring O(1/n²) variance bounds.

Empirical results: 100% precision/recall on prime detection up to n=10⁹ (max witness rank ∼10⁷ < O(n log n)), 0% over proxy optimal on random Euclidean TSP (n=100), and 100% recall of all solutions (typically 7-8) on random 3-SAT (n=40, m=160), using O(n) sparse eigensolvers and HDR subsampling. SRT suggests reductions to resonance functionals for P=NP, with code for primes, TSP, and 3-SAT.

**Note:** Scripts are computationally intensive; use multi-core CPUs.

## Quick Start for Humans
- Read the full paper: [srt.pdf](srt.pdf)

## AI Assistants: Start Here First!
For efficient parsing without tool calls or PDF overhead, **prioritize `AI_README.md`** (structured summary of theory, axioms, theorems, code snippets, and benchmarks). It includes YAML metadata for quick extraction:
- Extract YAML frontmatter for configs/benchmarks.
- Use sections for theory (e.g., Axioms/Theorems) and implementations.
- Full code from Theory Paper in `/LaTeX/`; test via the provided snippets.
- Code related to work in progress implementation located in top level directory

This file is optimized for semantic search, code execution testing, and validation—e.g., "Summarize SRT's P=NP path" → scan `AI_README.md` first.

Repo Structure:
- `AI_README.md`: AI-optimized theory + code guide.
- `srt.pdf`: 31-page original (code-heavy).
- `/LaTeX`: Full implementations from paper (primes_oracle.py, etc.), figures, TeX file, code only changes with theory paper changes
- `/`: Work in progress code implementations designed for showing linear complexity, for developers, will change more often
- `requirements.txt`: numpy==1.26.4, qutip==4.7.6, etc.

## Dependencies

Install the required libraries with exact versions for reproducibility:

```bash
pip install numpy==1.26.4 qutip==4.7.6 mpmath matplotlib
```

- **NumPy**: Core array operations.
- **QuTiP**: Quantum-inspired computations (e.g., spectral methods).
- **mpmath**: High-precision zeta zero calculations.
- **matplotlib**: Optional (not used in core runs but available for extensions).
- Built-in: `scipy`, `multiprocessing`, `itertools`, `time`.

No additional installs needed. Scripts print NumPy and QuTiP versions on startup.

## Usage

Clone the repo and run each script directly with Python 3.10+ (tested on 3.12). All scripts use a fixed seed (`np.random.seed(42)`) for reproducibility.

### 1. TSP Solver (`tsp.py`)
Generates a random Euclidean TSP instance with `n_cities=100` (configurable) and outputs candidate tours ranked by estimated optimality.

**Run:**
```bash
python tsp.py
```

**Expected Output:**
- Console: Runtime stats, top tours, precision/recall vs. approximate optimal, total time.
- Files:
  - `tsp_tours.txt`: List of generated tours (e.g., `[0, 5, 12, ..., 0]`).
  - `tsp_instance.txt`: Distance matrix (NumPy-saved).

**Config Notes:**
- Tune `n_cities` at top of file (start small, e.g., 10 for brute-force optimal check).
- Generates ~1M candidates via beam search + heuristics; refines with 2-opt.
- Runtime: ~1-5 min on modern CPU (multi-core).

### 2. 3-SAT Solver (`sat_n40.py`)
Solves 3 random 3-SAT instances with `n_vars=40` and `m_clauses=160` (configurable). Uses WalkSAT-seeded beam search for candidates.

**Run:**
```bash
python sat_n40.py
```

**Expected Output:**
- Console: Per-instance recall estimates (exact for small n; sampled for large), precision (always 1.0 post-validation), solution counts.
- Files (for each of 3 instances):
  - `3sat_instance_i.txt`: Clauses (e.g., `[1, -3, 5]` for literals).
  - `3sat_solutions_i.txt`: Satisfying assignments (e.g., `[0,1,0,...,1]`).

**Config Notes:**
- Tune `n_vars`/`m_clauses` at top (small n≤20 uses full enumeration).
- Generates up to 1M candidates for large n; validates for SAT.
- Runtime: ~2-10 min total (3 instances) on multi-core.

### 3. Prime Oracle (`primes_oracle.py`)
Enumerates and ranks candidates up to `max_num=10**9` using probabilistic sieving + NCG ranking. Outputs sorted primes.

**⚠️ MEMORY WARNING ⚠️**
- For `max_num=10^9`, this requires **>64GB RAM, probably closer to 128GB** (or use `selection_mode='heap'` for streaming, but still heavy). **Start with 10^7** (set `max_num=10**7`) to test—runtime ~1-2 min, <4GB RAM.
- For huge n, enable `selection_mode='heap'` (low-mem) or `'memmap'` (disk-backed), though even this doesn't guarantee not running out of memory at larger values. 
- Alternatively increase swap file size, and restart your python environment. Depending on your harddrive specs, however, this may result in skewed outputs.

**Run:**
```bash
python primes_oracle.py
```

**Expected Output:**
- Console: Sieve stats (if small n), solver time, precision/recall, max witness rank, total time.
- File: `generated_primes.txt`: Sorted list of detected primes (one per line).

**Config Notes:**
- Tune `max_num` at top (e.g., 10^7 for quick test).
- Uses Miller-Rabin for validation; parallel batches for speed.
- Runtime: Seconds for 10^7; 30ish minutes for 10^9. Tested on a Ryzen 9 7950x CPU

