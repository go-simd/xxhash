//go:build ignore

// Command gen produces accum_riscv64.s with go-asmgen: the XXH3 stripe
// accumulator on RVV (accumStripeRVV / accumRunRVV / accumScrambleRVV).
//
// One stripe is eight uint64 lanes (VL = 8, SEW = 64). For every lane:
//
//	dk  = dv ^ key
//	lo  = dk & 0xFFFFFFFF          (dk << 32 >> 32)
//	hi  = dk >> 32
//	acc += lo*hi                   (low 64 bits = full product, since lo,hi<2^32)
//	acc += swap-adjacent-lane(dv)  (the acc[i^1] += dv mapping)
//
// The adjacent-lane swap uses a gather with index vid^1 = [1,0,3,2,5,4,7,6].
// Everything is endian-neutral integer add/mul, so the digest matches every
// other architecture. Requires V with VLEN >= 128 (VL=8 at SEW=64 needs 512-bit
// LMUL=4; we set LMUL=M4).
//
// The input stripe (p) and secret (sec) are loaded as byte elements (vle8.v),
// not vle64.v: xxhash feeds arbitrarily-aligned input, and a vle64.v of an
// 8-byte-misaligned address SIGBUSes on hardware lacking misaligned vector
// support (observed on the SpacemiT X60). Reinterpreting the byte-loaded
// register group as u64 lanes is a no-op on this little-endian target. acc is a
// *[8]uint64 and is always 8-byte aligned, so its vle64.v stays.
//
// Three shapes are emitted, mirroring the amd64 kernel:
//
//   - accumStripeRVV(acc, p, sec): fold a single stripe (the final overlapping
//     tail stripe).
//   - accumRunRVV(acc, p, sec, nStripes): fold nStripes *consecutive* stripes in
//     one call, keeping the eight accumulator lanes resident in the V28 group
//     across the whole run (loaded once, stored once) and advancing the secret
//     by 8 bytes per stripe (the secret[s*8:] schedule). The accumulator is no
//     longer round-tripped through memory per stripe, so the mul/add dependency
//     chain stops stalling on a per-stripe load/store and Go/asm boundary.
//   - accumScrambleRVV(acc, p, sec, nBlocks): fold nBlocks complete 1024-byte
//     blocks (16 stripes + the inter-block scramble each) in one call, with the
//     accumulator resident across the WHOLE multi-block run and the scramble run
//     in-register (v ^= v>>47; v ^= secret[128:192]; v *= prime32_1 — a single
//     VMULVX since RVV has a native 64-bit vector multiply).
//
// Run: go run accum_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func stripeSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec")},
		[]abi.Arg{},
	)
}

func runSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{
			abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec"),
			abi.Scalar("nStripes", abi.Int64),
		},
		[]abi.Arg{},
	)
}

func blockSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{
			abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec"),
			abi.Scalar("nBlocks", abi.Int64),
		},
		[]abi.Arg{},
	)
}

// stripeBody folds one stripe into the accumulator group V28, reading the input
// from the address in X6 and the secret from X7. acc (V28) is assumed already
// loaded and stays resident. The vid^1 gather index is recomputed each call (it
// is cheap and keeps the body self-contained). Scratch vector groups: V0, V4,
// V8, V12, V16, V20, V24. Clobbers X8, X9.
func stripeBody(b *riscv64.Builder) {
	b.
		Raw("MOV $64, X9").
		Raw("VSETVLI X9, E8, M4, TA, MA, X8").  // 64 byte elements
		Raw("VLE8V (X6), V8").                  // dv bytes (alignment-safe)
		Raw("VLE8V (X7), V12").                 // key bytes (alignment-safe)
		Raw("VSETVLI $8, E64, M4, TA, MA, X8"). // reinterpret: 8 lanes of u64
		Raw("MOV $32, X9").
		Raw("VXORVV V8, V12, V16"). // dk = dv ^ key
		// lo = dk & 0xFFFFFFFF via << 32 >> 32 ; hi = dk >> 32
		Raw("VSLLVX X9, V16, V20").
		Raw("VSRLVX X9, V20, V20").  // lo
		Raw("VSRLVX X9, V16, V24").  // hi
		Raw("VMULVV V20, V24, V20"). // lo*hi (full 64-bit product)
		Raw("VADDVV V28, V20, V28"). // acc += mul
		// swapped dv: idx = vid ^ 1, gather
		Raw("VIDV V4").
		Raw("VXORVI $1, V4, V4").
		Raw("VRGATHERVV V4, V8, V0"). // V0[i] = dv[idx[i]]
		Raw("VADDVV V28, V0, V28")    // acc += swap(dv)
}

func main() {
	f := emit.NewFile("riscv64")

	// --- single stripe ---
	b := riscv64.NewFunc("accumStripeRVV", stripeSig(), 0)
	b.LoadArg("acc", "X5").LoadArg("p_base", "X6").LoadArg("sec_base", "X7").
		Raw("MOV $8, X9").
		Raw("VSETVLI X9, E64, M4, TA, MA, X8").
		Raw("VLE64V (X5), V28")
	stripeBody(b)
	b.Raw("VSE64V V28, (X5)").Ret()
	f.Add(b.Func())

	// --- multi-stripe run: acc resident in V28 across nStripes stripes. ---
	// X6 = p, X7 = sec, X10 = stripe counter. Each iteration advances p by 64 and
	// sec by 8 (the secret[s*8:] schedule).
	vr := riscv64.NewFunc("accumRunRVV", runSig(), 0)
	vr.LoadArg("acc", "X5").LoadArg("p_base", "X6").LoadArg("sec_base", "X7").
		LoadArg("nStripes", "X10").
		Raw("MOV $8, X9").
		Raw("VSETVLI X9, E64, M4, TA, MA, X8").
		Raw("VLE64V (X5), V28").
		Label("accumRunRVV_loop")
	stripeBody(vr)
	vr.
		Raw("ADD $64, X6, X6").
		Raw("ADD $8, X7, X7").
		Raw("ADD $-1, X10, X10").
		Raw("BNEZ X10, accumRunRVV_loop").
		Raw("VSE64V V28, (X5)").
		Ret()
	f.Add(vr.Func())

	// --- full-block run: nBlocks complete 1024-byte blocks (16 stripes +
	// scramble) in one call, acc resident in V28 across the WHOLE run. X6 = p
	// (advances continuously), X12 = secret base (fixed), X7 = working secret
	// cursor (reset to X12 at the top of each block), X10 = block counter. The
	// 16 stripes are emitted explicitly (X6 += 64, X7 += 8 per stripe). ---
	vb := riscv64.NewFunc("accumScrambleRVV", blockSig(), 0)
	vb.LoadArg("acc", "X5").LoadArg("p_base", "X6").LoadArg("sec_base", "X12").
		LoadArg("nBlocks", "X10").
		Raw("MOV $8, X9").
		Raw("VSETVLI X9, E64, M4, TA, MA, X8").
		Raw("VLE64V (X5), V28").
		// prime32_1 scalar in X11 for the scramble multiply.
		Raw("MOV $0x9E3779B1, X11").
		Label("accumScrambleRVV_loop").
		Raw("MOV X12, X7") // reset secret cursor to the block's base
	for s := 0; s < 16; s++ {
		stripeBody(vb)
		vb.
			Raw("ADD $64, X6, X6").
			Raw("ADD $8, X7, X7")
	}
	// In-register scramble: v ^= v>>47; v ^= secret[128:192]; v *= prime32_1.
	// X12 is the secret base, so the scramble secret chunk is at X12+128.
	vb.
		Raw("MOV $47, X9").
		Raw("VSRLVX X9, V28, V24").  // v >> 47
		Raw("VXORVV V28, V24, V28"). // v ^= v>>47
		Raw("ADD $128, X12, X13").
		Raw("MOV $64, X9").
		Raw("VSETVLI X9, E8, M4, TA, MA, X8").  // 64 secret bytes
		Raw("VLE8V (X13), V20").                // secret[128:192] as bytes (alignment-safe)
		Raw("VSETVLI $8, E64, M4, TA, MA, X8"). // reinterpret as 8 u64 lanes
		Raw("VXORVV V28, V20, V28").            // v ^= secret
		Raw("VMULVX X11, V28, V28").            // v *= prime32_1 (native 64-bit mul)
		Raw("ADD $-1, X10, X10").
		Raw("BNEZ X10, accumScrambleRVV_loop").
		Raw("VSE64V V28, (X5)").
		Ret()
	f.Add(vb.Func())

	if err := os.WriteFile("accum_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_riscv64.s")
}
