//go:build ignore

// Command gen produces accum_ppc64le.s with go-asmgen: the XXH3 stripe
// accumulator on VSX (accumStripeVSX / accumRunVSX / accumScrambleVSX).
//
// VSX registers are 128-bit = two uint64 lanes, so a 64-byte stripe is four
// 16-byte chunks. For each chunk:
//
//	dk  = dv ^ key
//	hi  = dk >> 32                   (VSRD by 32, per doubleword)
//	lo  = (dk << 32) >> 32           (low 32 bits of dk, zero-extended)
//	acc += lo*hi                     via VMULOUW: POWER8/9 have no 64-bit vector
//	                                 multiply, but lo and hi each occupy the odd
//	                                 word of their doubleword, and VMULOUW
//	                                 multiplies the odd words, producing the full
//	                                 64-bit product (lo,hi < 2^32, so it is exact).
//	acc += swap-doublewords(dv)      (VSLDOI $8 exchanges the two lanes = the
//	                                 acc[i^1] += dv mapping)
//
// Loads use LXVD2X (and stores STXVD2X): unlike the byte-wise LXVB16X, LXVD2X
// yields each lane as a natural 64-bit element value, which is what the
// per-doubleword VSRD/VSLD shifts and the VMULOUW need. VSX<->AltiVec aliasing:
// LXVD2X writes a VS register and VS(32+n) is the same physical register as
// AltiVec Vn, so we load into VS32.. and operate as V0..
//
// Three shapes are emitted, mirroring the amd64 kernel:
//
//   - accumStripeVSX(acc, p, sec): fold a single stripe (the final overlapping
//     tail stripe).
//   - accumRunVSX(acc, p, sec, nStripes): fold nStripes *consecutive* stripes in
//     one call, keeping the four accumulator pairs resident in V24..V27 across
//     the whole run (loaded once, stored once) and advancing the secret by 8
//     bytes per stripe (the secret[s*8:] schedule). The accumulator pairs are no
//     longer round-tripped through memory per chunk, so the mul/add chains
//     software-pipeline instead of stalling on a per-stripe load/store and
//     Go/asm boundary — this is the lever that closed the ~0.49x ppc64le gap.
//   - accumScrambleVSX(acc, p, sec, nBlocks): fold nBlocks complete 1024-byte
//     blocks (16 stripes + the inter-block scramble each) in one call, with the
//     accumulator resident across the WHOLE run and the scramble run in-register
//     (v ^= v>>47; v ^= secret[128+i*8]; v *= prime32_1 — full 64x32->64 via two
//     VMULOUW on the odd words, the same structure as amd64's two PMULUDQ).
//
// All math is endian-neutral integer add/shift/mul, so the digest is identical
// on every architecture; the exact operand orders are pinned by the qemu
// official-vector and differential tests. Only baseline VSX ops (LXVD2X / VSRD /
// VSLD / VMULOUW / VADDUDM / VSLDOI / XXLXOR) are used, so the kernel runs on
// POWER8 as well as POWER9 (no ISA-3.0 op, no dispatch gate needed).
//
// Run: go run accum_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
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

// accReg[c] holds the c-th accumulator pair (lanes 2c, 2c+1) resident across a
// run. Its VS alias is VS(32+n).
var accReg = []string{"V24", "V25", "V26", "V27"}

func accVS(c int) int { return 32 + 24 + c } // VS register number for accReg[c]

// shiftSetup loads the per-doubleword shift count 32 into V11 (VS43).
func shiftSetup(b *ppc64.Builder) {
	b.
		Raw("MOVD $32, R6").
		Raw("MTVSRD R6, VS43").
		Raw("XXPERMDI VS43, VS43, $0, VS43") // splat doubleword 0 to both
}

// stripeBody folds the four chunks of one stripe into the resident accumulator
// pairs accReg, reading dv from R4 (offset 16*c) and the secret from R5
// (offset 16*c). Scratch: V0/VS32 (dv), V1/VS33 (key), V2 (dk), V3, V4, V5, V6.
func stripeBody(b *ppc64.Builder) {
	for c := 0; c < 4; c++ {
		off := c * 16
		acc := accReg[c]
		b.
			Raw("MOVD $%d, R6", off).
			Raw("LXVD2X (R4)(R6), VS32").   // V0 = dv
			Raw("LXVD2X (R5)(R6), VS33").   // V1 = key
			Raw("XXLXOR VS32, VS33, VS34"). // V2 = dk = dv ^ key
			Raw("VSRD V2, V11, V4").        // V4 = hi = dk >> 32
			Raw("VSLD V2, V11, V3").        // V3 = dk << 32
			Raw("VSRD V3, V11, V3").        // V3 = lo = low 32 of dk
			Raw("VMULOUW V3, V4, V5").      // V5 = lo*hi per doubleword
			Raw("VSLDOI $8, V0, V0, V6").   // V6 = swap the two lanes of dv
			Raw("VADDUDM " + acc + ", V5, " + acc).
			Raw("VADDUDM " + acc + ", V6, " + acc)
	}
}

// loadAcc / storeAcc move the four accumulator pairs between memory (R3) and the
// resident V24..V27. R6 is the index register.
func loadAcc(b *ppc64.Builder) {
	for c := 0; c < 4; c++ {
		b.Raw("MOVD $%d, R6", c*16).Raw("LXVD2X (R3)(R6), VS%d", accVS(c))
	}
}
func storeAcc(b *ppc64.Builder) {
	for c := 0; c < 4; c++ {
		b.Raw("MOVD $%d, R6", c*16).Raw("STXVD2X VS%d, (R3)(R6)", accVS(c))
	}
}

func main() {
	f := emit.NewFile("ppc64le")

	// --- single stripe ---
	b := ppc64.NewFunc("accumStripeVSX", stripeSig(), 0)
	b.LoadArg("acc", "R3").LoadArg("p_base", "R4").LoadArg("sec_base", "R5")
	shiftSetup(b)
	loadAcc(b)
	stripeBody(b)
	storeAcc(b)
	b.Ret()
	f.Add(b.Func())

	// --- multi-stripe run: acc resident in V24..V27 across nStripes stripes. ---
	// R4 = p, R5 = sec, R7 = stripe counter. Each iteration advances p by 64 and
	// sec by 8 (the secret[s*8:] schedule).
	vr := ppc64.NewFunc("accumRunVSX", runSig(), 0)
	vr.LoadArg("acc", "R3").LoadArg("p_base", "R4").LoadArg("sec_base", "R5").
		LoadArg("nStripes", "R7")
	shiftSetup(vr)
	loadAcc(vr)
	vr.Label("accumRunVSX_loop")
	stripeBody(vr)
	vr.
		Raw("ADD $64, R4").
		Raw("ADD $8, R5").
		Raw("ADD $-1, R7").
		Raw("CMP R7, $0").
		Raw("BNE accumRunVSX_loop")
	storeAcc(vr)
	vr.Ret()
	f.Add(vr.Func())

	// --- full-block run: nBlocks complete 1024-byte blocks (16 stripes +
	// scramble) in one call, acc resident across the WHOLE run. R4 = p (advances
	// continuously), R8 = secret base (fixed), R5 = working secret cursor (reset
	// to R8 at the top of each block), R7 = block counter. V12 = prime32_1
	// broadcast into the odd word of each doubleword for the scramble multiply. ---
	vb := ppc64.NewFunc("accumScrambleVSX", blockSig(), 0)
	vb.LoadArg("acc", "R3").LoadArg("p_base", "R4").LoadArg("sec_base", "R8").
		LoadArg("nBlocks", "R7")
	shiftSetup(vb)
	loadAcc(vb)
	// prime32_1 into the odd 32-bit word of each doubleword of V12 (VS44). MTVSRD
	// places the 64-bit GPR into the high doubleword; XXPERMDI $0 splats it to
	// both doublewords. prime32_1 < 2^32 so it sits in the low (odd) word.
	vb.
		Raw("MOVD $0x9E3779B1, R6").
		Raw("MTVSRD R6, VS44").
		Raw("XXPERMDI VS44, VS44, $0, VS44").
		Label("accumScrambleVSX_loop").
		Raw("MOVD R8, R5") // reset secret cursor to the block base
	for s := 0; s < 16; s++ {
		stripeBody(vb)
		vb.Raw("ADD $64, R4").Raw("ADD $8, R5")
	}
	// In-register scramble of the four pairs:
	//   v ^= v>>47; v ^= secret[128+c*16]; v *= prime32_1.
	// v *= prime32_1 (full 64x32->64): with v.lo32 and v.hi32 each in the odd
	// word, lo*prime (VMULOUW) + (hi*prime << 32) (VMULOUW then VSLD by 32).
	vb.
		Raw("MOVD $47, R6").
		Raw("MTVSRD R6, VS45").
		Raw("XXPERMDI VS45, VS45, $0, VS45") // V13 = [47,47]
	for c := 0; c < 4; c++ {
		acc := accReg[c]
		scrOff := 128 + c*16
		vb.
			Raw("VSRD "+acc+", V13, V2").             // V2 = v >> 47
			Raw("XXLXOR VS%d, VS34, VS34", accVS(c)). // V2 = v ^ (v>>47)
			Raw("MOVD $%d, R6", scrOff).
			Raw("LXVD2X (R8)(R6), VS35").   // V3 = secret chunk
			Raw("XXLXOR VS34, VS35, VS34"). // V2 = v ^ secret
			// lo32(v) in odd word: (v<<32)>>32 ; hi32(v) in odd word: v>>32.
			Raw("VSLD V2, V11, V4").      // V4 = v << 32
			Raw("VSRD V4, V11, V4").      // V4 = lo32(v) (odd word)
			Raw("VSRD V2, V11, V5").      // V5 = hi32(v) (odd word)
			Raw("VMULOUW V4, V12, V6").   // V6 = lo32(v)*prime
			Raw("VMULOUW V5, V12, V5").   // V5 = hi32(v)*prime
			Raw("VSLD V5, V11, V5").      // V5 = (hi32(v)*prime) << 32
			Raw("VADDUDM V6, V5, " + acc) // v = lo*prime + (hi*prime)<<32
	}
	// R4 already advanced by 16*64 = 1024 across the stripe loop (next block).
	vb.
		Raw("ADD $-1, R7").
		Raw("CMP R7, $0").
		Raw("BNE accumScrambleVSX_loop")
	storeAcc(vb)
	vb.Ret()
	f.Add(vb.Func())

	if err := os.WriteFile("accum_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_ppc64le.s")
}
