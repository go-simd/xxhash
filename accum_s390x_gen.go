//go:build ignore

// Command gen produces accum_s390x.s with go-asmgen: the XXH3 single-stripe
// accumulator on the z/Architecture vector facility (accumStripeVEC). The vector
// facility is baseline on z13+, so there is no runtime dispatch.
//
// s390x is the only BIG-ENDIAN target. XXH3 reads its input little-endian, but
// VL loads bytes in memory order into big-endian elements, so a freshly loaded
// doubleword holds the byte-reversed value. We therefore byte-reverse each
// 8-byte lane right after loading (VPERM with a per-doubleword reversal selector)
// so the vector lanes carry the true little-endian values — exactly what the Go
// scalar path obtains via binary.LittleEndian. From there the math is identical
// to every other architecture and lane-neutral.
//
// Per 16-byte chunk (two uint64 lanes):
//
//	dk  = dv ^ key
//	hi  = dk >> 32                 (VESRLG $32, per doubleword)
//	lo  = (dk << 32) >> 32         (low 32 bits of dk)
//	acc += lo*hi                   (VMLOF: odd-word unsigned widening multiply
//	                               32x32->64; the low 32 bits of each doubleword
//	                               are the odd fullword in big-endian element
//	                               order, lo,hi < 2^32 so it is exact)
//	acc += swap-lanes(dv)          (VPDI $4 exchanges the two doublewords = the
//	                               acc[i^1] += dv mapping)
//
// The accumulator is a Go [8]uint64; VST writes lane VALUES in the host's native
// big-endian layout, which is how Go reads the uint64s back, so only the INPUT
// needs the byte reversal. Bit-exactness (this is the big-endian risk) is the
// gate pinned by the qemu official-vector and differential tests.
//
// Run: go run accum_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec")},
		[]abi.Arg{},
	)
	b := s390x.NewFunc("accumStripeVEC", sig, 0)
	b.LoadArg("acc", "R1").LoadArg("p_base", "R2").LoadArg("sec_base", "R3").
		// V31 = per-doubleword byte-reversal selector for VPERM:
		// out byte k = in byte (rev within each 8-byte half).
		// indices: 7 6 5 4 3 2 1 0  15 14 13 12 11 10 9 8
		Raw("MOVD $revsel<>(SB), R5").
		Raw("VL (R5), V31")

	for c := 0; c < 4; c++ {
		off := c * 16
		b.
			Raw("VL %d(R2), V0", off).    // dv (raw, big-endian elements)
			Raw("VL %d(R3), V1", off).    // key
			Raw("VPERM V0, V0, V31, V0"). // byte-reverse each lane -> true LE values
			Raw("VPERM V1, V1, V31, V1").
			Raw("VX V0, V1, V2").      // dk = dv ^ key
			Raw("VESRLG $32, V2, V4"). // hi = dk >> 32
			Raw("VESLG $32, V2, V3").  // dk << 32
			Raw("VESRLG $32, V3, V3"). // lo = low 32 of dk
			Raw("VMLOF V3, V4, V5").   // lo*hi per doubleword (odd-word widening; the
			//                            low 32 bits of each doubleword are the odd
			//                            fullword in big-endian element order)
			Raw("VPDI $4, V0, V0, V6"). // swap the two lanes of dv
			Raw("VL %d(R1), V7", off).  // acc pair
			Raw("VAG V7, V5, V7").
			Raw("VAG V7, V6, V7").
			Raw("VST V7, %d(R1)", off)
	}
	b.Ret()

	f := emit.NewFile("s390x")
	f.Add(b.Func())
	out := f.String()
	// Append the reversal selector constant used by VPERM.
	out += "\nGLOBL revsel<>(SB), RODATA|NOPTR, $16\n"
	for i := 0; i < 16; i++ {
		// lane-local byte reversal: position i takes source byte (i&^7)|(7-(i&7)).
		src := (i &^ 7) | (7 - (i & 7))
		out += fmt.Sprintf("DATA revsel<>+%d(SB)/1, $%d\n", i, src)
	}
	if err := os.WriteFile("accum_s390x.s", []byte(out), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_s390x.s")
}
