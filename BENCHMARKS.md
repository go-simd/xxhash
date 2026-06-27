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
* arm64 (Apple NEON) numbers are captured below; the amd64 AVX2 column above is
  the GitHub Actions measurement. Different hardware/ISA rows are not directly
  comparable in absolute terms.

## arm64 (NEON, Apple — register-resident run + in-asm scramble)

**Methodology.** Local Apple-silicon NEON, Go stable, single core, `-count=2..3`,
ours vs `zeebo/xxh3` back-to-back on the same CPU. "Before" = the per-stripe
kernel (accumulator round-tripped to memory once per 64-byte stripe, one Go/asm
call per stripe); "after" = the `accumRun` + `accumScramble` kernels (the four
NEON accumulator pairs stay resident across the whole multi-block run and the
inter-block scramble runs in-register, mirroring the amd64 full-block kernel).

| size   | before (MB/s) | after (MB/s) | zeebo (MB/s) | ×zeebo before | ×zeebo after |
|--------|--------------:|-------------:|-------------:|--------------:|-------------:|
| 256 B  |  7092 |  9346 | 13900 | 0.51× | 0.67× |
| 1 KiB  | 11024 | 19700 | 17900 | 0.62× | **1.10×** |
| 8 KiB  | 12482 | 25400 | 18360 | 0.68× | **1.38×** |
| 64 KiB | 12586 | 26550 | 18700 | 0.66× | **1.42×** |

> **arm64 now beats zeebo on every long input (1 KiB+): ~1.1–1.4×, up from
> ~0.66×.** The register-residency + in-asm-scramble lever removed the per-stripe
> accumulator memory round-trip and the per-stripe Go/asm boundary that capped the
> kernel. The 256 B case (`hashLong` with only three tail stripes, no full block)
> improved 0.51×→0.67× but still trails — that residual is the short/mid finalize
> path shared with the scalar code, not the stripe loop.

## ppc64le / loong64 / riscv64 / s390x (cross-arch lever)

The same `accumRun` + `accumScramble` register-resident kernels were ported to
all four remaining SIMD arches (VSX, LSX, RVV, z/vector). The structural change
is identical to arm64 — accumulator pairs/lanes resident across the whole
multi-block run, scramble in-register (full 64×32→64 via two odd-word multiplies
on ppc64le/s390x; a native 64-bit `VMULV`/`VMULVX` on loong64/riscv64).

### riscv64 (RVV 1.0, real SpacemiT X60 silicon — cfarm95, MEASURED)

**Methodology.** cfarm95 (SpacemiT X60, **RVV 1.0**, `cpu.RISCV64.HasV == true`
verified — the RVV kernel ran, not the scalar fallback), Go 1.26.4, single core,
`-count=3`, best-of-3, ours vs `zeebo/xxh3` back-to-back on the same CPU. This is
the arch whose kernel previously hit the misaligned `vle64.v` SIGBUS (fixed in
`bc34657` by byte-loading the input stripe): **`TestReference` + `TestOfficialVectors`
+ the full suite pass byte-exact on real X60 silicon — no SIGBUS, no SIGILL.**

| size   | ours (MB/s) | zeebo (MB/s) | ×zeebo |
|--------|------------:|-------------:|-------:|
| 8 B    |    63.6 |  119.1 | 0.53× |
| 64 B   |   123.3 |  228.3 | 0.54× |
| 256 B  |   187.2 |  141.7 | **1.32×** |
| 1 KiB  |   341.7 |  136.6 | **2.50×** |
| 8 KiB  |   438.1 |  133.0 | **3.29×** |
| 64 KiB |   472.6 |  133.1 | **3.55×** |

> **riscv64 RVV crushes zeebo on every long input (256 B+): 1.3×–3.6×.** zeebo/xxh3
> has no RVV kernel — it runs the scalar path on RISC-V (flat ~130–230 MB/s) — so
> the register-resident RVV `accumScramble` kernel pulls ahead by 3.5× at 64 KiB
> (472 MB/s vs 133). The 8 B / 64 B cases trail (0.53×) because that path is the
> short-input merge/avalanche shared with scalar code, not the RVV stripe loop.
> Byte-exact and crash-free on real X60.

### ppc64le (VSX, real POWER9 + POWER8E silicon — cfarm433/cfarm112)

**Correctness MEASURED on real silicon:** `TestReference` + `TestOfficialVectors`
+ full suite pass byte-exact on cfarm433 (**POWER9**, VSX kernel active) and
cfarm112 (**POWER8E**, baseline-VSX path) — no SIGILL on the older ISA. Throughput
was sampled on cfarm433 but it is a heavily-shared 64-thread box, so absolute
numbers are noisy; best-of-3 at 64 KiB was ~4.07 GB/s ours vs ~4.49 GB/s zeebo
(**~0.91×**, near parity), narrowing from the prior ~0.49× single-stripe gap. The
ratio is indicative only given the shared host; correctness is the firm result.

### loong64 / s390x (qemu-verified)

loong64 la464 (LSX + scalar fallback) and s390x big-endian (official vectors +
reference + streaming) are verified under qemu on every length class — including
the new POWER8 ISA-guard and multi-VLEN (128/256/512) RVV CI lanes. cfarm401
(loong64) is air-gapped, and there is no z/Architecture cfarm node, so these two
rely on the qemu CI lanes for correctness; the riscv64 measured column above is
the directly-measured proxy for the identical cross-arch lever. 100% coverage on
all four arches.
