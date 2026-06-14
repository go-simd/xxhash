//go:build ignore

// Command gen produces accum_ppc64le.s with go-asmgen: the XXH3 single-stripe
// accumulator on VSX (accumStripeVSX).
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
//	                                 64-bit product (lo,hi < 2^32, so it is
//	                                 exact).
//	acc += swap-doublewords(dv)      (VSLDOI $8 exchanges the two lanes = the
//	                                 acc[i^1] += dv mapping)
//
// Loads use LXVD2X (and stores STXVD2X): unlike the byte-wise LXVB16X, LXVD2X
// yields each lane as a natural 64-bit element value, which is what the
// per-doubleword VSRD/VSLD shifts and the VMULOUW need. The shift mnemonics are
// "VSRD Vsrc, Vamt, Vdst" (value first, shift-count second). VSX<->AltiVec
// aliasing: LXVD2X writes a VS register and VS(32+n) is the same physical
// register as AltiVec Vn, so we load into VS32.. and operate as V0..
//
// All math is endian-neutral integer add/shift/mul, so the digest is identical
// on every architecture; the exact operand orders are pinned by the qemu
// official-vector and differential tests.
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

func main() {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec")},
		[]abi.Arg{},
	)
	b := ppc64.NewFunc("accumStripeVSX", sig, 0)
	b.LoadArg("acc", "R3").LoadArg("p_base", "R4").LoadArg("sec_base", "R5").
		// V11 = [32,32]: per-doubleword shift count.
		Raw("MOVD $32, R6").
		Raw("MTVSRD R6, VS43").
		Raw("XXPERMDI VS43, VS43, $0, VS43") // splat doubleword 0 to both

	for c := 0; c < 4; c++ {
		off := c * 16
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
			Raw("LXVD2X (R3)(R6), VS39").   // V7 = acc pair
			Raw("VADDUDM V7, V5, V7").
			Raw("VADDUDM V7, V6, V7").
			Raw("STXVD2X VS39, (R3)(R6)")
	}
	b.Ret()

	f := emit.NewFile("ppc64le")
	f.Add(b.Func())
	if err := os.WriteFile("accum_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_ppc64le.s")
}
