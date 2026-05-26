# hrp-bench-bun

Bun implementation of the **HRP (Hierarchical Risk Parity)** core, part of a
cross-language benchmark suite. The same algorithm is implemented in
[C](https://github.com/suenot/hrp-bench-c),
[C++](https://github.com/suenot/hrp-bench-cpp),
[Zig](https://github.com/suenot/hrp-bench-zig),
[Rust](https://github.com/suenot/hrp-bench-rust),
[Python](https://github.com/suenot/hrp-bench-python),
[Node.js](https://github.com/suenot/hrp-bench-node) and
[Bun](https://github.com/suenot/hrp-bench-bun).

The code is identical to the Node.js version (Bun runs the same `bench.mjs`); the
two repos exist to benchmark the two JavaScript runtimes separately. The
synthetic dataset uses the same 64-bit LCG as the C reference (via `BigInt`), so
prices and HRP weights are bit-identical across languages. Generation is not timed.

## Run

```sh
bun bench.mjs        # or: bun run bench
```

## What it measures

Five timed stages on `N` assets × 365 daily observations: log returns O(N·T),
covariance **O(N²·T, dominates)**, average linkage O(N²) (NN-chain), quasi-diagonalization
O(N²), recursive-bisection weights O(N log N). Correlation/distance are computed
but not timed. Sizes above 4 GB of `N×N` matrix are skipped.
