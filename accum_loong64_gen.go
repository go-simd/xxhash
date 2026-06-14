//go:build ignore

// Command gen produces accum_loong64.s with go-asmgen: the XXH3 single-stripe
// accumulator on LSX (accumStripeLSX).
//
// LSX registers are 128-bit = two uint64 lanes, so a 64-byte stripe is four V
// registers (V1..V4 hold the accumulator pairs). For each chunk:
//
//	dk  = dv ^ key
//	lo  = dk & 0xFFFFFFFF        (dk << 32 >> 32)
//	hi  = dk >> 32
//	acc += lo*hi                 (VMULV low 64 = full product, lo,hi < 2^32)
//	acc += swap-pair(dv)         (VSHUF4IW $0x4e swaps the two doublewords: it
//	                             reorders the four words [w0,w1,w2,w3] -> [w2,w3,w0,w1])
//
// All integer add/mul, lane-neutral, so the digest matches every architecture.
//
// Run: go run accum_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

func main() {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("acc", abi.Int64), abi.Slice("p"), abi.Slice("sec")},
		[]abi.Arg{},
	)
	b := loong64.NewFunc("accumStripeLSX", sig, 0)
	b.LoadArg("acc", "R4").LoadArg("p_base", "R5").LoadArg("sec_base", "R6")

	for c := 0; c < 4; c++ {
		off := c * 16
		b.
			Raw("MOVV $%d, R7", off).
			Raw("VMOVQ (R5)(R7), V5"). // dv
			Raw("VMOVQ (R6)(R7), V6"). // key
			Raw("VXORV V5, V6, V6").   // dk = dv ^ key
			Raw("VSLLV $32, V6, V7").
			Raw("VSRLV $32, V7, V7").      // lo = dk & 0xFFFFFFFF
			Raw("VSRLV $32, V6, V8").      // hi = dk >> 32
			Raw("VMULV V7, V8, V7").       // lo*hi
			Raw("VSHUF4IW $0x4e, V5, V9"). // swap the two u64 of dv: words [2,3,0,1]
			Raw("VMOVQ (R4)(R7), V10").    // acc pair
			Raw("VADDV V10, V7, V10").
			Raw("VADDV V10, V9, V10").
			Raw("VMOVQ V10, (R4)(R7)")
	}
	b.Ret()

	f := emit.NewFile("loong64")
	f.Add(b.Func())
	if err := os.WriteFile("accum_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote accum_loong64.s")
}
