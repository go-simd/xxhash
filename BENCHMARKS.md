# Performance parity — go-simd/xxhash (XXH3-64) vs zeebo/xxh3

**Algorithm.** go-simd/xxhash's `Sum64` is **XXH3 (64-bit)**, not XXH64 — its
kernels are the XXH3 single-stripe accumulator (amd64 AVX2, arm64 NEON,
ppc64le/s390x via go-asmgen). The fair same-algorithm reference is therefore
**`github.com/zeebo/xxh3`** (`xxh3.Hash`), the established pure-Go XXH3 SIMD
library go-simd/xxhash is already byte-verified against in `TestReference` — not
`cespare/xxhash`, which is the *different* XXH64 algorithm. Inputs 8 B … 64 KiB,
single core, `b.SetBytes(len)` so `go test` reports MB/s.

## amd64 (AVX2, GitHub Actions x86_64 runner — ratios valid, absolute ns/op CI-noisy)

**Methodology.** GitHub Actions `ubuntu-latest` runner, **AMD EPYC 7763** (`avx2`
present, **no `avx512*`** — confirmed from `/proc/cpuinfo`), `GOAMD64` baseline,
Go stable, single core. `-count=6`, **min-of-6**. The runner is shared, so
absolute throughput is noisy; the **ratio vs zeebo/xxh3** is measured
back-to-back on the *same* CPU and is valid. Reproduce via
`gh workflow run bench-amd64.yml`.

| size | go-simd (MB/s) | zeebo/xxh3 (ref) | ×zeebo | verdict |
|------|---------------:|-----------------:|-------:|---------|
| 8 B    | 1829 |  2333 | 0.78× | trails |
| 64 B   | 9327 | 12235 | 0.76× | trails |
| 256 B  | 4341 | 12074 | 0.36× | trails |
| 1 KiB  | 7840 | 30491 | 0.26× | trails |
| 8 KiB  | 9188 | 49593 | 0.19× | trails |
| 64 KiB | 9510 | 53389 | 0.18× | **trails (~5.6×)** |

> **Honest finding (amd64): go-simd/xxhash does NOT yet beat the reference.**
> On amd64 it reaches only **0.18–0.78× of zeebo/xxh3**, the gap widening with
> size (zeebo sustains ~53 GB/s at 64 KiB vs go-simd's ~9.5 GB/s). The cause is
> structural: go-simd runs a **single-stripe** XXH3 accumulator, while zeebo runs
> the full **multi-accumulator (8-lane) AVX2** XXH3 with software-pipelined
> stripe processing — so zeebo hides latency and keeps more vector ALUs busy.
> go-simd's kernel is correct (byte-exact to zeebo, fuzz-verified) but
> throughput-limited by the single-accumulator dependency chain.

### Action items
1. **Multi-accumulator kernel:** widen the XXH3 kernel from one stripe to the
   canonical 8-lane accumulator bank (and unroll/pipeline the stripe loop) so the
   amd64 path stops being latency-bound. This is the path to closing the ~5.6×
   gap to zeebo at large inputs.
2. Re-measure on AVX-512 silicon once available (the GitHub Actions runner here
   is AVX2-only) — zeebo gains substantially from AVX-512, and the parity bar
   should be set against the same ISA.

### Notes
* `Sum64` is bit-exact to `xxh3.Hash` on every length class incl. path/block
  boundaries (`TestReference`, 100% coverage, fuzz-clean). The shortfall is
  throughput, not correctness.
* arm64 (M4 Max NEON) numbers are not yet captured in this file; the amd64 AVX2
  column above is the GitHub Actions measurement. Different hardware/ISA rows are
  not directly comparable in absolute terms.
