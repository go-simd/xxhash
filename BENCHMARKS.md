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

The table tracks three kernel generations, all measured on the **same EPYC 7763**
runner: **(a)** the original single-stripe kernel (accumulator round-tripped
through memory once per 64-byte stripe); **(b)** the multi-accumulator run
kernel (accumulator banks resident across a run of stripes, but the per-block
scramble still in scalar Go and a Go/asm call per block); and **(c)** the
**full-block** kernel (a single asm call folds every 1024-byte block — 16
fully-unrolled stripes **plus the inter-block scramble in-register** — with the
accumulator resident across the whole multi-block run and `PREFETCHT0` ahead of
use; the long-input path now crosses the Go/asm boundary just once). Ratios are
min-of-6 vs `zeebo/xxh3` back-to-back on the same CPU.

| size | (a) single | (b) multi-acc | (c) full-block | zeebo/xxh3 | ×zeebo (b) | ×zeebo (c) |
|------|-----------:|--------------:|---------------:|-----------:|-----------:|-----------:|
| 8 B    | 1829 |  1833 |  1834 |  2332 | 0.79× | 0.79× |
| 64 B   | 9327 |  9326 |  9328 | 12266 | 0.76× | 0.76× |
| 256 B  | 4341 |  5393 |  5433 | 12037 | 0.45× | 0.45× |
| 1 KiB  | 7840 | 16703 | 16562 | 30492 | 0.55× | 0.54× |
| 8 KiB  | 9188 | 26516 | 40928 | 49528 | 0.54× | **0.83×** |
| 64 KiB | 9510 | 29057 | 50266 | 53324 | 0.55× | **0.94×** |

> **Honest finding (amd64): the full-block kernel reaches near-parity at scale —
> 0.94× zeebo at 64 KiB (was 0.55×) and 0.83× at 8 KiB (was 0.54×).** Folding the
> 16-stripe block *and its scramble* into one in-register asm call — so the
> long-input path crosses the Go/asm boundary once for the whole input instead of
> twice per 1024-byte block, and the scramble runs vectorized instead of in
> scalar Go — was the dominant lever. 64 KiB throughput went from ~29 GB/s to
> **~50 GB/s** (a further **1.73×** on top of the multi-accumulator kernel, and
> **~5.3×** over the original single-stripe kernel), landing within **6%** of
> zeebo. The benefit grows with input size because the per-block savings amortize
> over more blocks: at 1 KiB (a single block) there is essentially nothing to
> amortize, so it holds the multi-accumulator kernel's ~0.54×; the remaining gap
> to zeebo there is the short-input merge/avalanche plus the one boundary
> crossing, not the stripe loop. The unroll-multiple-stripes-per-iter lever (the
> 16 stripes are already fully unrolled here) and prefetch are folded in.
> Correctness is unchanged — byte-exact to zeebo and the canonical golden vectors
> on every length class incl. block boundaries, fuzz-clean, 100% cover, verified
> on both AVX2 and SSE2 dispatch and on big-endian s390x.

### Action items
1. **Close the remaining 1 KiB gap (~0.54×):** at a single block the win is in
   the short-input merge and removing the lone boundary crossing, not the stripe
   loop — a small-long-input fast path could help.
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
