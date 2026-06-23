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

The table shows the kernel **before** (single-stripe: the accumulator round-tripped
through memory once per 64-byte stripe) and **after** the multi-accumulator
kernel (the four 256-bit accumulator banks stay resident in vector registers
across a whole run of stripes), both measured on the same EPYC 7763 runner.

| size | before (MB/s) | after (MB/s) | zeebo/xxh3 (ref) | ×zeebo before | ×zeebo after |
|------|--------------:|-------------:|-----------------:|--------------:|-------------:|
| 8 B    | 1829 |  1833 |  2327 | 0.78× | 0.79× |
| 64 B   | 9327 |  9326 | 12241 | 0.76× | 0.76× |
| 256 B  | 4341 |  5393 | 12015 | 0.36× | 0.45× |
| 1 KiB  | 7840 | 16703 | 30487 | 0.26× | **0.55×** |
| 8 KiB  | 9188 | 26516 | 49441 | 0.19× | **0.54×** |
| 64 KiB | 9510 | 29057 | 53267 | 0.18× | **0.55×** |

> **Honest finding (amd64): the multi-accumulator kernel closes most of the
> structural gap, but zeebo's hand-tuned asm still leads ~1.8× at scale.**
> The big-input throughput went from ~9.5 GB/s to **~29 GB/s** — a **~3.05×
> speedup at 64 KiB** — and the ratio vs zeebo went from **0.18× to 0.55×**.
> Crucially the gap no longer *widens* with size: the single-stripe version
> collapsed to 0.18× because the per-stripe load/store serialized the kernel;
> the multi-accumulator version holds a flat ~0.55× from 1 KiB to 64 KiB because
> the four independent mul/add chains now software-pipeline. It does **not** yet
> reach parity: zeebo additionally unrolls multiple stripes per loop iteration
> and folds the whole long-input pipeline (block scramble included) in assembly,
> avoiding go-simd's per-block Go/asm call boundary. Correctness is unchanged —
> byte-exact to zeebo and the canonical golden vectors, fuzz-clean, 100% cover.

### Action items
1. **Close the remaining ~1.8×:** unroll the stripe loop (process N stripes per
   iteration) and move the per-block scramble into the asm run so the long-input
   path never crosses the Go/asm boundary mid-block, matching zeebo's structure.
2. Re-measure on AVX-512 silicon once available (the GitHub Actions runner here
   is AVX2-only) — zeebo gains substantially from AVX-512, and the parity bar
   should be set against the same ISA.

### Notes
* `Sum64` is bit-exact to `xxh3.Hash` on every length class incl. path/block
  boundaries (`TestReference`, 100% coverage, fuzz-clean) — re-verified after the
  multi-accumulator kernel under qemu-x86_64 (both AVX2 and SSE2 dispatch paths)
  and qemu-s390x (big-endian). The shortfall is throughput, not correctness.
* arm64 (M4 Max NEON) numbers are not yet captured in this file; the amd64 AVX2
  column above is the GitHub Actions measurement. Different hardware/ISA rows are
  not directly comparable in absolute terms.
