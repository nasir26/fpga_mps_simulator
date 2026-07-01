# Running the benchmarks

- [`bench_ceiling.py`](bench_ceiling.py) — per-family ceiling-discovery driver
- [`test_scaling.py`](test_scaling.py) — shared circuit-family definitions (`CIRCUIT_FAMILIES`) and FPGA timeout helpers, imported by `bench_ceiling.py`
- [`extra_circuits.py`](extra_circuits.py) — W-State, GHZ-Ladder, TFIM-Trotter, Random-Brickwork
- [`benchmark.py`](benchmark.py) — separate MPS vs Aer MPS vs Aer SV timing comparison (see `make benchmark`)
- [`results/`](results/), [`results_highcap/`](results_highcap/) — output data + plots (see [../README.md](../README.md) and [../docs/DESIGN_NOTES.md](../docs/DESIGN_NOTES.md) for the findings)

## Method: coarse doubling → bisection

`bench_ceiling.py` discovers each family's qubit ceiling independently
(rather than sweeping every family over one fixed qubit list, which breaks
down once families have wildly different limits):

1. **Coarse growth** — try `[2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]`, then
   keep doubling (`2000, 4000, ...`) until a run hard-fails, exceeds
   `--timeout` seconds, or would exceed `--max-cap`.
2. **Bisection refine** (on by default) — binary search between the last
   passing size and the first failing size until the gap is `<= --tol`
   qubits, turning "passed 2000, failed 4000" into a precise ceiling.
3. Each family runs inside its own try/except, so one family crashing its
   driver logic can't abort the others.

## Run the full 9-family sweep

```bash
cd benchmarks
python3 bench_ceiling.py
```

Defaults: `--chi-max 64 --shots 1024 --timeout 60 --max-cap 8192 --bisect`.
Output goes to `results/` (`--out results`).

## Run a single family

```bash
python3 bench_ceiling.py --families TFIM-Trotter
```

## High-capacity re-run (for families still climbing at the default timeout)

Used for GHZ and 1Q-Rots, which were still passing when the 60s timeout cut
off the main sweep:

```bash
python3 bench_ceiling.py --families 1Q-Rots GHZ --timeout 300 --max-cap 4000 --out results_highcap
```

## Outputs

- `summary.csv` / `summary.json` — every attempt (all families)
- `ceilings.json` — `max_ok_qubits` / `first_fail_qubits` / failure reason per family
- `<family>_time.png`, `<family>_metrics.png` — per-family plots
- `combined_time.png`, `combined_bond.png`, `combined_throughput.png` — cross-family overlays
- `ceilings_bar.png` — sorted ceiling bar chart
