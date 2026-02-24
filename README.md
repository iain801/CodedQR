# CodedQR

Fault-tolerant parallel QR decomposition via **Coded Parallel Block Modified Gram-Schmidt (PBMGS)**. Checksum-based encoding is applied to both Q and R factors so that the decomposition can recover from node failures during distributed computation.

## Overview

Standard parallel QR decomposition on large HPC clusters is vulnerable to process failures. CodedQR addresses this by embedding algebraic checksums into the matrix factors before and during the PBMGS algorithm. When failures occur, the lost data is reconstructed by solving a linear system derived from the surviving checksums, followed by a post-orthogonalization step to restore orthogonality of Q.

**Key capabilities:**
- Parallel Block Modified Gram-Schmidt over a 2D process grid
- Horizontal checksums (R factor) and vertical checksums (Q factor) via generator matrices
- Configurable fault tolerance level `f` (up to `sqrt(np)/2` simultaneous failures)
- Failure simulation and reconstruction benchmarking
- Performance analysis and visualization tooling

## Project Structure

```
├── codedqr_main.c        # Entry point: setup, encoding, PBMGS, recovery, validation
├── codedqr_base.c        # Core routines: checksums, PBMGS, reconstruction, post-ortho
├── codedqr_base.h        # Public API and configuration flags
├── Makefile              # Build configuration (Intel MPI + MKL)
├── codedqr.job           # SLURM batch script (single job, parameter sweep)
├── codedqr-root.job      # SLURM root job (spawns sub-jobs)
├── codedqr-sub.job       # SLURM sub-job script
├── test-stats.py         # Generates performance plots from CSV data
├── test-plots.ipynb      # Jupyter notebook for interactive analysis
├── Prototypes/           # Earlier implementations (serial MGS, serial BMGS, etc.)
├── data/                 # Collected timing CSVs
└── figs/                 # Generated figures
```

## Requirements

- **Intel MPI** (`mpiicc`)
- **Intel MKL** (BLAS, LAPACK, VSL)
- **SLURM** (for cluster execution)
- **Python 3** with `pandas`, `matplotlib`, `numpy` (for analysis)

## Building

```bash
make codedqr
```

Produces `out/codedqr_main`. Requires `MKLROOT` to be set (e.g. via `module load mkl/latest`).

## Usage

```bash
mpirun -n <np> ./out/codedqr_main <n> <f> [log_file]
```

| Argument   | Description |
|------------|-------------|
| `np`       | Number of MPI processes (must be a perfect square) |
| `n`        | Global matrix dimension (n x n) |
| `f`        | Max tolerable faults (`f <= sqrt(np)/2`) |
| `log_file` | Optional CSV path for timing output |

The process count `np` should satisfy `np = (p + f)^2` where `p` is the desired grid side length and `f` is the fault tolerance level.

### Example

Run on a 10x10 process grid (100 processes) with fault tolerance for up to 2 failures on a 24000x24000 matrix:

```bash
mpirun -n 100 ./out/codedqr_main 24000 2 data/results.csv
```

### SLURM

```bash
sbatch codedqr.job
```

Runs a parameter sweep over matrix sizes, grid sizes, and fault levels, writing timing data to `data/`.

## Output

Each run prints timing breakdowns to stdout and optionally appends a CSV row:

```
p,n,f,recovery,final solve,post-ortho,cs construct,pbmgs
```

| Metric         | Description |
|----------------|-------------|
| `cs construct` | Checksum generator matrix construction + encoding time |
| `pbmgs`        | PBMGS decomposition time |
| `recovery`     | Failure detection + reconstruction time |
| `post-ortho`   | Post-orthogonalization time |
| `final solve`  | Optional Rx = Q'b solve time |

## Analysis

Generate plots from collected data:

```bash
python test-stats.py
```

Or use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook test-plots.ipynb
```

## Configuration

Compile-time flags in `codedqr_base.h`:

| Flag             | Default | Description |
|------------------|---------|-------------|
| `TEST_FAILIURE`  | 1       | Simulate node failures and test reconstruction |
| `DEBUG`          | 0       | Enable verbose debug output |
| `SET_SEED`       | 0       | Fixed RNG seed (0 = time-based) |
| `DO_FINAL_SOLVE` | 0      | Solve Rx = Q'b after decomposition |

## Author

Iain Weissburg, 2023
